#include "ResultRecordset.hpp"

namespace aaj = axis::application::jobs;
namespace aaocm = axis::application::output::collectors::messages;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace asmm = axis::services::messaging;
namespace asdi = axis::services::diagnostics::information;

aaor::ResultRecordset::~ResultRecordset(void)
{
  // nothing to do here
}

void aaor::ResultRecordset::OpenRecordset( const axis::String& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::CloseRecordset( void )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginCreateField( void )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::CreateField( const axis::String&, FieldType )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::CreateMatrixField( const axis::String&, int, int )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::CreateVectorField( const axis::String&, int )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndCreateField( void )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginAnalysisStep( const axis::String&, int , 
                                               const asdi::SolverCapabilities&)
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndAnalysisStep( void )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginSnapshot( const ada::AnalysisInfo& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndSnapshot( const ada::AnalysisInfo& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginNodeRecord( const asmm::ResultMessage&, const ade::Node& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndNodeRecord( const asmm::ResultMessage&, const ade::Node& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginElementRecord( const asmm::ResultMessage&, const ade::FiniteElement& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndElementRecord( const asmm::ResultMessage&, const ade::FiniteElement& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::BeginGenericRecord( const asmm::ResultMessage&, const ada::NumericalModel& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::EndGenericRecord( const asmm::ResultMessage&, const ada::NumericalModel& )
{
  // base implementation does nothing 
}

void aaor::ResultRecordset::Init( aaj::WorkFolder& workFolder )
{
  // base implementation does nothing 
}
