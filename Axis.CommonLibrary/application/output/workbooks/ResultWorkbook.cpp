#include "ResultWorkbook.hpp"
#include <utility>
#include "ResultWorkbook_Pimpl.hpp"
#include "../recordsets/ResultRecordset.hpp"
#include "foundation/InvalidOperationException.hpp"
#include "foundation/NotSupportedException.hpp"
#include "foundation/ArgumentException.hpp"
#include "foundation/NotImplementedException.hpp"
#include "application/jobs/AnalysisStepInformation.hpp"
#include "foundation/OutOfBoundsException.hpp"

namespace aaow = axis::application::output::workbooks;
namespace aaor = axis::application::output::recordsets;
namespace aaj = axis::application::jobs;
namespace ada = axis::domain::analyses;
namespace asdi = axis::services::diagnostics::information;
namespace af = axis::foundation;
namespace afd = axis::foundation::date_time;
namespace afu = axis::foundation::uuids;

aaow::ResultWorkbook::ResultWorkbook( void )
{
  pimpl_ = new Pimpl();
}

aaow::ResultWorkbook::~ResultWorkbook(void)
{
  // destroy all recordsets
  Pimpl::RecordsetSet::iterator end = pimpl_->AllRecordsets.end();
  for (Pimpl::RecordsetSet::iterator it = pimpl_->AllRecordsets.begin(); it != end; ++it)
  {
    aaor::ResultRecordset& r = **it;
    r.Destroy();
  }  
  pimpl_->AllRecordsets.clear();
  pimpl_->Elements.clear();
  pimpl_->Nodes.clear();
  delete pimpl_;
  pimpl_ = NULL;
}

void aaow::ResultWorkbook::InitWorkbook( aaj::WorkFolder& workFolder )
{
  DoBeforeInit(workFolder);
  if (SupportsMainRecordset())
  {
    pimpl_->MainRecordset = &DoCreateMainRecordset(workFolder);
  }
  if (SupportsGenericRecordset())
  {
    pimpl_->GenericRecordset = &DoCreateGenericRecordset(workFolder);
  }
  Pimpl::RecordsetSet::iterator end = pimpl_->AllRecordsets.end();
  for (Pimpl::RecordsetSet::iterator it = pimpl_->AllRecordsets.begin(); it != end; ++it)
  {
    aaor::ResultRecordset& r = **it;
    r.Init(workFolder);
    if (!r.IsInitialized())
    {
      throw axis::foundation::InvalidOperationException(_T("Recordset failed to initialize."));
    }
  }  
  DoAfterInit(workFolder);
}

void aaow::ResultWorkbook::Open(const aaj::AnalysisStepInformation& stepInfo)
{
  axis::String stepName = stepInfo.GetStepName();
  int stepIndex = stepInfo.GetStepIndex();
  asdi::SolverCapabilities& solverCaps = stepInfo.GetSolverCapabilities();

  DoBeforeOpen(stepInfo);
  // open main recordset first...
  if (pimpl_->MainRecordset != NULL) pimpl_->MainRecordset->OpenRecordset(_T(""));
  // ...open entity recordset afterwards...
  Pimpl::RecordsetList::iterator n_end = pimpl_->Nodes.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Nodes.begin(); it != n_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    r.OpenRecordset(it->first);
    if (!r.IsReady())
    {
      r.OpenRecordset(it->first);
      if (!r.IsReady()) // still not ready?
      {
        throw af::InvalidOperationException(_T("Could not open recordset."));
      }
    }
  }  
  Pimpl::RecordsetList::iterator e_end = pimpl_->Elements.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Elements.begin(); it != e_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    if (!r.IsReady())
    {
      r.OpenRecordset(it->first);
      if (!r.IsReady()) // still not ready?
      {
        throw af::InvalidOperationException(_T("Could not open recordset."));
      }
    }
  }  
  // ...and finally, open generic recordset
  if (pimpl_->GenericRecordset != NULL) pimpl_->GenericRecordset->OpenRecordset(_T(""));
    
  DoAfterOpen(stepInfo);
  pimpl_->IsOpen = true;
}

bool aaow::ResultWorkbook::IsOpen( void ) const
{
  return pimpl_->IsOpen && IsReady();
}

void aaow::ResultWorkbook::Close( void )
{
  // ignore if workbook is not open
  if (!pimpl_->IsOpen) return;

  DoBeforeClose();
  // close generic recordset first...
  if (pimpl_->GenericRecordset != NULL) 
  {
    pimpl_->GenericRecordset->CloseRecordset();
  }
  // ...close entity recordset afterwards...
  Pimpl::RecordsetList::iterator n_end = pimpl_->Nodes.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Nodes.begin(); it != n_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    if (r.IsReady()) // avoid closing the same thing over and over
    {
      r.CloseRecordset();
      if (r.IsReady()) // if recordset is bad-behaved...
      {
        throw af::InvalidOperationException(_T("Unexpected recordset behavior after closing it."));
      }
    }
  }  
  Pimpl::RecordsetList::iterator e_end = pimpl_->Elements.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Elements.begin(); it != e_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    if (r.IsReady()) // avoid closing the same thing over and over
    {
      r.CloseRecordset();
      if (r.IsReady()) // if recordset is bad-behaved...
      {
        throw af::InvalidOperationException(_T("Unexpected recordset behavior after closing it."));
      }
    }
  }  
  // ...per last, close main recordset
  if (pimpl_->MainRecordset != NULL)
  {
    pimpl_->MainRecordset->CloseRecordset();
  }
  DoAfterClose();
  pimpl_->IsOpen = false;
}

void aaow::ResultWorkbook::BeginStep( const aaj::AnalysisStepInformation& stepInfo )
{
  if (!pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException();
  }

  axis::String stepName = stepInfo.GetStepName();
  int stepIndex = stepInfo.GetStepIndex();
  asdi::SolverCapabilities& solverCaps = stepInfo.GetSolverCapabilities();

  if (pimpl_->MainRecordset != NULL) 
  {
    pimpl_->MainRecordset->BeginAnalysisStep(stepName, stepIndex, solverCaps);
  }
  Pimpl::RecordsetList::iterator n_end = pimpl_->Nodes.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Nodes.begin(); it != n_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    r.BeginAnalysisStep(stepName, stepIndex, solverCaps);
  }  
  Pimpl::RecordsetList::iterator e_end = pimpl_->Elements.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Elements.begin(); it != e_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    r.BeginAnalysisStep(stepName, stepIndex, solverCaps);
  }  
  if (pimpl_->GenericRecordset != NULL) 
  {
    pimpl_->GenericRecordset->BeginAnalysisStep(stepName, stepIndex, solverCaps);
  }
}

void aaow::ResultWorkbook::EndStep( void )
{
  if (!pimpl_->IsOpen)
  {
    throw axis::foundation::InvalidOperationException();
  }

  if (pimpl_->GenericRecordset != NULL) 
  {
    pimpl_->GenericRecordset->EndAnalysisStep();
  }
  Pimpl::RecordsetList::iterator n_end = pimpl_->Nodes.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Nodes.begin(); it != n_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    r.EndAnalysisStep();
  }  
  Pimpl::RecordsetList::iterator e_end = pimpl_->Elements.end();
  for (Pimpl::RecordsetList::iterator it = pimpl_->Elements.begin(); it != e_end; ++it)
  { 
    aaor::ResultRecordset& r = *(*it).second;
    r.EndAnalysisStep();
  }  
  if (pimpl_->MainRecordset != NULL)
  {
    pimpl_->MainRecordset->EndAnalysisStep();
  }
}

void aaow::ResultWorkbook::CreateNodeRecordset( const axis::String& nodeSetName )
{
  if (!SupportsNodeRecordset())
  {
    throw af::NotSupportedException();
  }
  if (IsOpen())
  {
    throw af::InvalidOperationException(_T("Cannot create recordset after database is opened."));
  }
  // check for duplicates
  if (pimpl_->NodeRecordsetExists(nodeSetName))
  {
    throw af::ArgumentException(_T("A recordset already exists for this entity."));
  }
  aaor::ResultRecordset& r = DoCreateNodeRecordset(nodeSetName); 
  pimpl_->Nodes.push_back(std::make_pair(nodeSetName, &r));
  if (!pimpl_->RecordsetExists(r)) // is this recordset unique (ie., new)?
  {
    pimpl_->AllRecordsets.insert(&r);
  }
}

void aaow::ResultWorkbook::CreateElementRecordset( const axis::String& elementSetName )
{
  if (!SupportsElementRecordset())
  {
    throw af::NotSupportedException();
  }
  if (IsOpen())
  {
    throw af::InvalidOperationException(_T("Cannot create recordset after database is opened."));
  }
  // check for duplicates
  if (pimpl_->ElementRecordsetExists(elementSetName))
  {
    throw af::ArgumentException(_T("A recordset already exists for this entity."));
  }
  aaor::ResultRecordset& r = DoCreateElementRecordset(elementSetName); 
  pimpl_->Elements.push_back(std::make_pair(elementSetName, &r));
  if (!pimpl_->RecordsetExists(r)) // is this recordset unique (ie., new)?
  {
    pimpl_->AllRecordsets.insert(&r);
  }
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetNodeRecordset( const axis::String& nodeSetName )
{
  if (!SupportsNodeRecordset())
  {
    throw af::NotSupportedException();
  }
  if (!pimpl_->NodeRecordsetExists(nodeSetName))
  {
    throw af::ArgumentException(_T("Recordset not found."));
  }
  return pimpl_->GetNodeRecordset(nodeSetName);
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetNodeRecordset( int index )
{
  if (!SupportsNodeRecordset())
  {
    throw af::NotSupportedException();
  }
  if (index >= GetNodeRecordsetCount())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *pimpl_->Nodes[index].second;
}

int aaow::ResultWorkbook::GetNodeRecordsetCount( void ) const
{
  return (int)pimpl_->Nodes.size();
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetElementRecordset( int index )
{
  if (!SupportsElementRecordset())
  {
    throw af::NotSupportedException();
  }
  if (index >= GetElementRecordsetCount())
  {
    throw axis::foundation::OutOfBoundsException();
  }
  return *pimpl_->Elements[index].second;
}

int aaow::ResultWorkbook::GetElementRecordsetCount( void ) const
{
  return (int)pimpl_->Elements.size();
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetElementRecordset( const axis::String& elementSetName )
{
  if (!SupportsElementRecordset())
  {
    throw af::NotSupportedException();
  }
  if (!pimpl_->ElementRecordsetExists(elementSetName))
  {
    throw af::ArgumentException(_T("Recordset not found."));
  }
  return pimpl_->GetElementRecordset(elementSetName);
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetGenericSetRecordset( void )
{
  if (!SupportsGenericRecordset())
  {
    throw af::NotSupportedException();
  }
  return *pimpl_->GenericRecordset;
}

aaor::ResultRecordset& aaow::ResultWorkbook::GetMainRecordset( void )
{
  if (!SupportsMainRecordset())
  {
    throw af::NotSupportedException();
  }
  return *pimpl_->MainRecordset;
}

void aaow::ResultWorkbook::BeginSnapshot( const ada::AnalysisInfo& info )
{
  DoBeforeBeginSnapshot(info);
  if (pimpl_->MainRecordset != NULL) pimpl_->MainRecordset->BeginSnapshot(info);
  Pimpl::RecordsetSet::iterator end = pimpl_->AllRecordsets.end();
  for (Pimpl::RecordsetSet::iterator it = pimpl_->AllRecordsets.begin(); it != end; ++it)
  {
    aaor::ResultRecordset& r = **it;
    r.BeginSnapshot(info);
  }
  if (pimpl_->GenericRecordset != NULL) pimpl_->GenericRecordset->BeginSnapshot(info);
  DoAfterBeginSnapshot(info);
}

void aaow::ResultWorkbook::EndSnapshot( const ada::AnalysisInfo& info )
{
  DoBeforeEndSnapshot(info);
  if (pimpl_->GenericRecordset != NULL) pimpl_->GenericRecordset->EndSnapshot(info);
  Pimpl::RecordsetSet::iterator end = pimpl_->AllRecordsets.end();
  for (Pimpl::RecordsetSet::iterator it = pimpl_->AllRecordsets.begin(); it != end; ++it)
  {
    aaor::ResultRecordset& r = **it;
    r.EndSnapshot(info);
  }
  if (pimpl_->MainRecordset != NULL) pimpl_->MainRecordset->EndSnapshot(info);
  DoAfterEndSnapshot(info);
}

/***************************** BASE IMPLEMENTATION METHODS *****************************/
void aaow::ResultWorkbook::DoBeforeInit( aaj::WorkFolder& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoAfterInit( aaj::WorkFolder& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoBeforeOpen( const aaj::AnalysisStepInformation& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoAfterOpen( const aaj::AnalysisStepInformation& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoBeforeClose( void )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoAfterClose( void )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoBeforeBeginSnapshot( const ada::AnalysisInfo& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoAfterBeginSnapshot( const ada::AnalysisInfo& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoBeforeEndSnapshot( const ada::AnalysisInfo& )
{
  // nothing to do in base implementation!
}

void aaow::ResultWorkbook::DoAfterEndSnapshot( const ada::AnalysisInfo& info )
{
  // nothing to do in base implementation!
}

aaor::ResultRecordset& aaow::ResultWorkbook::DoCreateNodeRecordset( const axis::String& )
{
  throw axis::foundation::NotImplementedException();
}

aaor::ResultRecordset& aaow::ResultWorkbook::DoCreateElementRecordset( const axis::String& )
{
  throw axis::foundation::NotImplementedException();
}

aaor::ResultRecordset& aaow::ResultWorkbook::DoCreateGenericRecordset( aaj::WorkFolder& )
{
  throw axis::foundation::NotImplementedException();
}

aaor::ResultRecordset& aaow::ResultWorkbook::DoCreateMainRecordset( aaj::WorkFolder& )
{
  throw axis::foundation::NotImplementedException();
}

bool aaow::ResultWorkbook::IsReady( void ) const
{
  return true;
}

axis::String aaow::ResultWorkbook::GetWorkbookOutputName( void ) const
{
  return pimpl_->OutputName;
}

void aaow::ResultWorkbook::SetWorkbookOutputName( const axis::String& name )
{
  if (IsOpen())
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot change name when output file is opened."));
  }
  pimpl_->OutputName = name;
}

void axis::application::output::workbooks::ResultWorkbook::ToggleAppendOperation( bool appendState )
{
  if (IsOpen())
  {
    throw axis::foundation::InvalidOperationException(_T("Cannot change append operation state when output is opened."));
  }
  if (!SupportsAppendOperation() && appendState)
  {
    throw axis::foundation::NotSupportedException();
  }
  pimpl_->IsAppend = appendState;
}

bool axis::application::output::workbooks::ResultWorkbook::IsAppendOperation( void ) const
{
  return pimpl_->IsAppend;
}
