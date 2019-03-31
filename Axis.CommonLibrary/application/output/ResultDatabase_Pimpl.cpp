#include "ResultDatabase_Pimpl.hpp"
#include "workbooks/ResultWorkbook.hpp"
#include "recordsets/ResultRecordset.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/collections/ElementSet.hpp"

namespace aao = axis::application::output;
namespace aaoc = axis::application::output::collectors;
namespace aaow = axis::application::output::workbooks;
namespace aaor = axis::application::output::recordsets;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade  = axis::domain::elements;
namespace asmm = axis::services::messaging;

namespace {
  typedef aao::CollectorChain<axis::domain::elements::Node, aaoc::NodeSetCollector> NodeSetCollectorList;
  typedef aao::CollectorChain<axis::domain::elements::FiniteElement, aaoc::ElementSetCollector> ElementSetCollectorList;
}

aao::ResultDatabase::Pimpl::Pimpl(void)
{
  CurrentWorkbook = NULL;
  IsOpen = false;
}

aao::ResultDatabase::Pimpl::~Pimpl(void)
{
  if (CurrentWorkbook != NULL)
  {
    if (CurrentWorkbook->IsOpen())
    {
      CurrentWorkbook->Close();
    }
    CurrentWorkbook->Destroy();
    CurrentWorkbook = NULL;
  }
}

void axis::application::output::ResultDatabase::Pimpl::PopulateAndInitWorkbook( axis::application::jobs::WorkFolder& workFolder )
{
  // init recordsets
  aaow::ResultWorkbook& w = *CurrentWorkbook;
  int nodeChainCount = NodeCollectors.GetChainCount();
  for (int i = 0; i < nodeChainCount; ++i)
  {
    w.CreateNodeRecordset(NodeCollectors[i].GetTargetSetName());
  }
  int elementChainCount = ElementCollectors.GetChainCount();
  for (int i = 0; i < elementChainCount; ++i)
  {
    w.CreateElementRecordset(ElementCollectors[i].GetTargetSetName());
  }
  w.InitWorkbook(workFolder);
}

void aao::ResultDatabase::Pimpl::WriteNodalResults( const asmm::ResultMessage& message, 
                                                    const ada::NumericalModel& numericalModel )
{
  aaow::ResultWorkbook& workbook = *CurrentWorkbook;

  int nodeChainCount = NodeCollectors.GetChainCount();
  for (int chainIdx = 0; chainIdx < nodeChainCount; ++chainIdx)
  {
    NodeSetCollectorList& list = NodeCollectors[chainIdx];
    if (list.IsOfInterest(message))
    {
      axis::String setName = list.GetTargetSetName();
      adc::NodeSet *nodeSet;
      if (setName.empty())
      {
        nodeSet = &numericalModel.Nodes();
      }
      else
      {
        nodeSet = &numericalModel.GetNodeSet(setName);
      }    
      aaor::ResultRecordset& recordset = workbook.GetNodeRecordset(setName);

      for (size_type nodeIdx = 0; nodeIdx < nodeSet->Count(); ++nodeIdx)
      {
        const ade::Node& node = (*nodeSet).GetByPosition(nodeIdx);
        recordset.BeginNodeRecord(message, node);
        list.ChainCollect(message, recordset, node);
        recordset.EndNodeRecord(message, node);
      }    
    }
  }
}

void aao::ResultDatabase::Pimpl::WriteElementResults( const asmm::ResultMessage& message, 
                                                      const ada::NumericalModel& numericalModel )
{
  Pimpl::ElementSetCollectorGroup& elementChains = ElementCollectors;
  aaow::ResultWorkbook& workbook = *CurrentWorkbook;

  int nodeChainCount = elementChains.GetChainCount();
  for (int chainIdx = 0; chainIdx < nodeChainCount; ++chainIdx)
  {
    ElementSetCollectorList& list = elementChains[chainIdx];
    if (list.IsOfInterest(message))
    {
      axis::String setName = list.GetTargetSetName();
      adc::ElementSet *elementSet;
      if (setName.empty())
      {
        elementSet = &numericalModel.Elements();
      }
      else
      {
        elementSet = &numericalModel.GetElementSet(setName);
      }    
      aaor::ResultRecordset& recordset = workbook.GetElementRecordset(setName);

      for (size_type elementIdx = 0; elementIdx < elementSet->Count(); ++elementIdx)
      {
        const ade::FiniteElement& element = (*elementSet).GetByPosition(elementIdx);
        recordset.BeginElementRecord(message, element);
        list.ChainCollect(message, recordset, element);
        recordset.EndElementRecord(message, element);
      }    
    }
  }
}

void aao::ResultDatabase::Pimpl::WriteGenericResults( const asmm::ResultMessage& message, 
                                                      const ada::NumericalModel& numericalModel )
{
  if (GenericCollectors.empty()) return;
  aaow::ResultWorkbook& workbook = *CurrentWorkbook;
  aaor::ResultRecordset& recordset = workbook.GetGenericSetRecordset();
  recordset.BeginGenericRecord(message, numericalModel);
  Pimpl::GenericCollectorSet::iterator end = GenericCollectors.end();
  for (Pimpl::GenericCollectorSet::iterator it = GenericCollectors.begin(); it != end; ++it)
  {
    (**it).Collect(message, recordset, numericalModel);
  }
  recordset.EndGenericRecord(message, numericalModel);
}

void axis::application::output::ResultDatabase::Pimpl::CreateRecordsetFields( void )
{
  CreateNodeRecordsetFields();
  CreateElementRecordsetFields();
  
  // generic and main recordsets don't hold fields
  if (CurrentWorkbook->SupportsGenericRecordset())
  {
    aaor::ResultRecordset& gr = CurrentWorkbook->GetGenericSetRecordset();
    gr.BeginCreateField(); gr.EndCreateField();
  }
  if (CurrentWorkbook->SupportsMainRecordset())
  {
    aaor::ResultRecordset& mr = CurrentWorkbook->GetMainRecordset();
    mr.BeginCreateField(); mr.EndCreateField();
  }
}

void axis::application::output::ResultDatabase::Pimpl::CreateNodeRecordsetFields( void )
{
  aaow::ResultWorkbook& w = *CurrentWorkbook;
  int nodeChainCount = NodeCollectors.GetChainCount();
  for (int nodeChainIdx = 0; nodeChainIdx < nodeChainCount; ++nodeChainIdx)
  {
    aaor::ResultRecordset& rs = w.GetNodeRecordset(NodeCollectors[nodeChainIdx].GetTargetSetName());
    NodeSetCollectorList& ncl = NodeCollectors[nodeChainIdx];
    rs.BeginCreateField();
    for (int collectorIdx = 0; collectorIdx < ncl.GetCollectorCount(); ++collectorIdx)
    {
      const aaoc::NodeSetCollector& collector = ncl[collectorIdx];
      DataType type = collector.GetFieldType();
      switch (type)
      {
      case axis::application::output::kMatrix:
        rs.CreateMatrixField(collector.GetFieldName(), 
          collector.GetMatrixFieldRowCount(), 
          collector.GetMatrixFieldColumnCount());
        break;
      case axis::application::output::kVector:
        rs.CreateVectorField(collector.GetFieldName(), collector.GetVectorFieldLength());
        break;
      default:
        rs.CreateField(collector.GetFieldName(), (aaor::ResultRecordset::FieldType)type);
        break;
      }
    }
    rs.EndCreateField();
  }
}

void axis::application::output::ResultDatabase::Pimpl::CreateElementRecordsetFields( void )
{
  aaow::ResultWorkbook& w = *CurrentWorkbook;
  int elementChainCount = ElementCollectors.GetChainCount();
  for (int elementChainIdx = 0; elementChainIdx < elementChainCount; ++elementChainIdx)
  {
    aaor::ResultRecordset& rs = w.GetElementRecordset(ElementCollectors[elementChainIdx].GetTargetSetName());
    ElementSetCollectorList& ecl = ElementCollectors[elementChainIdx];
    rs.BeginCreateField();
    for (int collectorIdx = 0; collectorIdx < ecl.GetCollectorCount(); ++collectorIdx)
    {
      const aaoc::ElementSetCollector& collector = ecl[collectorIdx];
      DataType type = collector.GetFieldType();
      switch (type)
      {
      case axis::application::output::kMatrix:
        rs.CreateMatrixField(collector.GetFieldName(), 
                             collector.GetMatrixFieldRowCount(), 
                             collector.GetMatrixFieldColumnCount());
        break;
      case axis::application::output::kVector:
        rs.CreateVectorField(collector.GetFieldName(), collector.GetVectorFieldLength());
        break;
      default:
        rs.CreateField(collector.GetFieldName(), (aaor::ResultRecordset::FieldType)type);
        break;
      }
    }
    rs.EndCreateField();
  }
}

void axis::application::output::ResultDatabase::Pimpl::SetUpAllCollectors( void )
{
  CollectorSet::iterator end = AllCollectors.end();
  for (CollectorSet::iterator it = AllCollectors.begin(); it != end; ++it)
  {
    aaoc::EntityCollector& c = **it;
    c.Prepare();
  }
}

void axis::application::output::ResultDatabase::Pimpl::TearDownAllCollectors( void )
{
  CollectorSet::iterator end = AllCollectors.end();
  for (CollectorSet::iterator it = AllCollectors.begin(); it != end; ++it)
  {
    aaoc::EntityCollector& c = **it;
    c.TearDown();
  }
}
