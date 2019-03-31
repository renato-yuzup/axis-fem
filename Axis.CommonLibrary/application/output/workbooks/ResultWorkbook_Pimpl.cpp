#include "ResultWorkbook_Pimpl.hpp"
#include "foundation/InvalidOperationException.hpp"

namespace aaor = axis::application::output::recordsets;
namespace aaow = axis::application::output::workbooks;


aaow::ResultWorkbook::Pimpl::Pimpl( void )
{
  IsOpen = false;
  IsAppend = false;
  GenericRecordset = NULL;
  MainRecordset = NULL;
}

aaow::ResultWorkbook::Pimpl::~Pimpl( void )
{
  // nothing to do here
}

bool aaow::ResultWorkbook::Pimpl::NodeRecordsetExists( const axis::String& nodeSetName ) const
{
  RecordsetList::const_iterator end = Nodes.end();
  for (RecordsetList::const_iterator it = Nodes.begin(); it != end; ++it)
  {
    if (it->first == nodeSetName) return true;
  }
  return false;
}

bool aaow::ResultWorkbook::Pimpl::ElementRecordsetExists( const axis::String& elementSetName ) const
{
  RecordsetList::const_iterator end = Elements.end();
  for (RecordsetList::const_iterator it = Elements.begin(); it != end; ++it)
  {
    if (it->first == elementSetName) return true;
  }
  return false;
}

bool aaow::ResultWorkbook::Pimpl::RecordsetExists( const aaor::ResultRecordset& recordset ) const
{
  aaor::ResultRecordset *r = &const_cast<aaor::ResultRecordset&>(recordset);
  return AllRecordsets.find(r) != AllRecordsets.end();
}

aaor::ResultRecordset& aaow::ResultWorkbook::Pimpl::GetNodeRecordset( const axis::String& nodeSetName )
{
  RecordsetList::const_iterator end = Nodes.end();
  for (RecordsetList::const_iterator it = Nodes.begin(); it != end; ++it)
  {
    if (it->first == nodeSetName) return *(it->second);
  }
  throw axis::foundation::InvalidOperationException();
}

aaor::ResultRecordset& aaow::ResultWorkbook::Pimpl::GetElementRecordset( const axis::String& elementSetName )
{
  RecordsetList::const_iterator end = Elements.end();
  for (RecordsetList::const_iterator it = Elements.begin(); it != end; ++it)
  {
    if (it->first == elementSetName) return *(it->second);
  }
  throw axis::foundation::InvalidOperationException();
}
