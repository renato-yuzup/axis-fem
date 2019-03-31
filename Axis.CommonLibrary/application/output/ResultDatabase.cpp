#include "ResultDatabase.hpp"
#include "ResultDatabase_Pimpl.hpp"
#include "workbooks/ResultWorkbook.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/ElementNotFoundException.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "domain/algorithms/messages/SnapshotStartMessage.hpp"
#include "domain/algorithms/messages/SnapshotEndMessage.hpp"

namespace aaj = axis::application::jobs;
namespace aao  = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace aaocm = axis::application::output::collectors::messages;
namespace aaor = axis::application::output::recordsets;
namespace aaow = axis::application::output::workbooks;
namespace ada  = axis::domain::analyses;
namespace adam = axis::domain::algorithms::messages;
namespace ade  = axis::domain::elements;
namespace asdi = axis::services::diagnostics::information;
namespace asmm = axis::services::messaging;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;


aao::ResultDatabase::ResultDatabase(void)
{
  pimpl_ = new Pimpl();
}

aao::ResultDatabase::~ResultDatabase(void)
{
  delete pimpl_;
  pimpl_ = NULL;
}

void aao::ResultDatabase::Destroy( void ) const
{
  delete this;
}

void aao::ResultDatabase::OpenDatabase( aaj::WorkFolder& workFolder )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Database is already open."));
  }
  pimpl_->PopulateAndInitWorkbook(workFolder);
  pimpl_->CreateRecordsetFields();
  pimpl_->IsOpen = true;
}

void aao::ResultDatabase::CloseDatabase( void )
{
  if (!pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Database is already closed."));
  }
  pimpl_->CurrentWorkbook->Close();
  pimpl_->IsOpen = false;
}

bool aao::ResultDatabase::IsOpen( void ) const
{
  return pimpl_->IsOpen;
}

bool aao::ResultDatabase::HasCollectors( void ) const
{
  return !pimpl_->AllCollectors.empty();
}

void aao::ResultDatabase::StartStep(const aaj::AnalysisStepInformation& stepInfo)
{
  pimpl_->CurrentWorkbook->Open(stepInfo);
  pimpl_->CurrentWorkbook->BeginStep(stepInfo);
  pimpl_->SetUpAllCollectors();
}

void aao::ResultDatabase::EndStep( void )
{
  pimpl_->TearDownAllCollectors();
  pimpl_->CurrentWorkbook->EndStep();
}

void aao::ResultDatabase::StartSnapshot( const adam::SnapshotStartMessage& message )
{
  pimpl_->CurrentWorkbook->BeginSnapshot(message.GetAnalysisInformation());
}

void aao::ResultDatabase::EndSnapshot( const adam::SnapshotEndMessage& message )
{
  pimpl_->CurrentWorkbook->EndSnapshot(message.GetAnalysisInformation());
}

axis::String aao::ResultDatabase::GetFormatTitle( void ) const
{
  if (pimpl_->CurrentWorkbook == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return pimpl_->CurrentWorkbook->GetFormatTitle();
}

axis::String aao::ResultDatabase::GetOutputFileName( void ) const
{
  if (pimpl_->CurrentWorkbook == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return pimpl_->CurrentWorkbook->GetWorkbookOutputName();
}

axis::String aao::ResultDatabase::GetFormatDescription( void ) const
{
  if (pimpl_->CurrentWorkbook == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return pimpl_->CurrentWorkbook->GetShortDescription();
}

bool axis::application::output::ResultDatabase::GetAppendState( void ) const
{
  if (pimpl_->CurrentWorkbook == NULL)
  {
    throw axis::foundation::InvalidOperationException();
  }
  return pimpl_->CurrentWorkbook->IsAppendOperation();
}

int aao::ResultDatabase::GetCollectorCount( void ) const
{
  return (int)pimpl_->AllCollectors.size();
}

const aaoc::EntityCollector& aao::ResultDatabase::operator[]( int index ) const
{
  return *pimpl_->AllCollectors[index];
}

const aaoc::EntityCollector& aao::ResultDatabase::GetCollector( int index ) const
{
  return operator [](index);
}

void aao::ResultDatabase::AddNodeCollector( aaoc::NodeSetCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  pimpl_->NodeCollectors.AddCollector(collector);
  pimpl_->AllCollectors.push_back(&collector);
}

void aao::ResultDatabase::AddElementCollector( aaoc::ElementSetCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  pimpl_->ElementCollectors.AddCollector(collector);
  pimpl_->AllCollectors.push_back(&collector);
}

void aao::ResultDatabase::AddGenericCollector( aaoc::GenericCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  Pimpl::GenericCollectorSet::iterator end = pimpl_->GenericCollectors.end();
  for (Pimpl::GenericCollectorSet::iterator it = pimpl_->GenericCollectors.begin(); it != end; ++it)
  {
    if (*it == &collector)
    {
      throw axis::foundation::ArgumentException(_T("Duplicate not allowed."));
    }
  }
  pimpl_->GenericCollectors.push_back(&collector);
  pimpl_->AllCollectors.push_back(&collector);
}

void aao::ResultDatabase::RemoveNodeCollector( aaoc::NodeSetCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  pimpl_->NodeCollectors.RemoveCollector(collector);
  Pimpl::CollectorSet::iterator end = pimpl_->AllCollectors.end();
  for (Pimpl::CollectorSet::iterator it = pimpl_->AllCollectors.begin(); it != end; ++it)
  {
    if (*it == &collector)
    {
      pimpl_->AllCollectors.erase(it);
      break;
    }
  }
}

void aao::ResultDatabase::RemoveElementCollector( aaoc::ElementSetCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  pimpl_->ElementCollectors.RemoveCollector(collector);
  Pimpl::CollectorSet::iterator end = pimpl_->AllCollectors.end();
  for (Pimpl::CollectorSet::iterator it = pimpl_->AllCollectors.begin(); it != end; ++it)
  {
    if (*it == &collector)
    {
      pimpl_->AllCollectors.erase(it);
      break;
    }
  }
}

void aao::ResultDatabase::RemoveGenericCollector( aaoc::GenericCollector& collector )
{
  if (pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot modify database when it is opened."));
  }
  Pimpl::GenericCollectorSet::iterator end = pimpl_->GenericCollectors.end();
  for (Pimpl::GenericCollectorSet::iterator it = pimpl_->GenericCollectors.begin(); it != end; ++it)
  {
    if (*it == &collector)
    {
      pimpl_->GenericCollectors.erase(it);
      Pimpl::CollectorSet::iterator all_end = pimpl_->AllCollectors.end();
      for (Pimpl::CollectorSet::iterator all_it = pimpl_->AllCollectors.begin(); all_it != all_end; ++all_it)
      {
        if (*all_it == &collector)
        {
          pimpl_->AllCollectors.erase(all_it);
          break;
        }
      }
      return;
    }
  }
  throw axis::foundation::ElementNotFoundException();
}

void aao::ResultDatabase::SetWorkbook( aaow::ResultWorkbook& workbook )
{
  pimpl_->CurrentWorkbook = &workbook;
}

void aao::ResultDatabase::WriteResults( const asmm::ResultMessage& message, 
                                        const ada::NumericalModel& numericalModel )
{
  if (!pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException(_T("Database not ready."));
  }
  pimpl_->WriteNodalResults(message, numericalModel);
  pimpl_->WriteElementResults(message, numericalModel);
  pimpl_->WriteGenericResults(message, numericalModel);
}
