#include "ResultBucketConcrete.hpp"
#include "ResultBucketConcrete_Pimpl.hpp"
#include "application/output/ResultDatabase.hpp"
#include "application/output/ChainMetadata.hpp"
#include "application/output/collectors/EntityCollector.hpp"
#include "application/output/collectors/messages/AnalysisStepStartMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepEndMessage.hpp"
#include "domain/algorithms/messages/SnapshotStartMessage.hpp"
#include "domain/algorithms/messages/SnapshotEndMessage.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace aaocm = axis::application::output::collectors::messages;
namespace adam = axis::domain::algorithms::messages;
namespace aaj = axis::application::jobs;
namespace aao = axis::application::output;
namespace asmm = axis::services::messaging;
namespace ada = axis::domain::analyses;

aao::ResultBucketConcrete::ResultBucketConcrete( void )
{
  pimpl_ = new Pimpl();
}

aao::ResultBucketConcrete::~ResultBucketConcrete( void )
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aao::ResultBucketConcrete::Destroy( void ) const
{
  delete this;
}

void aao::ResultBucketConcrete::AddDatabase( ResultDatabase& database )
{
  Pimpl::database_list::iterator end = pimpl_->Databases.end();
  for (Pimpl::database_list::iterator it = pimpl_->Databases.begin(); it != end; ++it)
  {
    if (*it == &database)
    {
      throw axis::foundation::ArgumentException(_T("Database already registered."));
    }
  }
  pimpl_->Databases.push_back(&database);
}

void aao::ResultBucketConcrete::RemoveDatabase( ResultDatabase& database )
{
  Pimpl::database_list::iterator end = pimpl_->Databases.end();
  for (Pimpl::database_list::iterator it = pimpl_->Databases.begin(); it != end; ++it)
  {
    if (*it == &database)
    {
      pimpl_->Databases.erase(it);
    }
  }
  throw axis::foundation::ElementNotFoundException(_T("Database not registered."));
}

void aao::ResultBucketConcrete::PlaceResult( const asmm::ResultMessage& message, const ada::NumericalModel& numericalModel )
{
  Pimpl::database_list::iterator end = pimpl_->Databases.end();
  for (Pimpl::database_list::iterator it = pimpl_->Databases.begin(); it != end; ++it)
  {
    ResultDatabase& db = **it;
    if (aaocm::AnalysisStepStartMessage::IsOfKind(message))
    { // ABRIR STEP
      const aaocm::AnalysisStepStartMessage& msg = static_cast<const aaocm::AnalysisStepStartMessage&>(message);
      aaj::AnalysisStepInformation stepInfo = msg.GetStepInformation();
      if (!db.IsOpen()) db.OpenDatabase(stepInfo.GetJobWorkFolder());
      db.StartStep(stepInfo);
    }
    else if (aaocm::AnalysisStepEndMessage::IsOfKind(message))
    { // FECHAR STEP
      const aaocm::AnalysisStepEndMessage& msg = static_cast<const aaocm::AnalysisStepEndMessage&>(message);
      db.EndStep();
      if (db.IsOpen()) db.CloseDatabase();
    }
    else if (adam::SnapshotStartMessage::IsOfKind(message))
    { // ABRIR SNAPSHOT
      const adam::SnapshotStartMessage& msg = static_cast<const adam::SnapshotStartMessage&>(message);
      db.StartSnapshot(msg);
    }
    else if (adam::SnapshotEndMessage::IsOfKind(message))
    { // FECHAR SNAPSHOT
      const adam::SnapshotEndMessage& msg = static_cast<const adam::SnapshotEndMessage&>(message);
      db.EndSnapshot(msg);
    }
    else 
    { // a common result message; just forward
      db.WriteResults(message, numericalModel);
    }
  }
}

aao::ChainMetadata aao::ResultBucketConcrete::GetChainMetadata( int index ) const
{
  if (index < 0 || index >= GetChainCount())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  ResultDatabase& db = *pimpl_->Databases[index];
  ChainMetadata metadata(db.GetFormatTitle(), db.GetOutputFileName(), db.GetFormatDescription());
  metadata.SetAppendDataState(db.GetAppendState());
  int count = db.GetCollectorCount();
  for (int i = 0; i < count; ++i)
  {
    metadata.AddCollectorDescription(db[i].GetFriendlyDescription());
  }
  return metadata;
}

int aao::ResultBucketConcrete::GetChainCount( void ) const
{
  return (int)pimpl_->Databases.size();
}
