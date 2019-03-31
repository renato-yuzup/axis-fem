#pragma once
#include "ResultWorkbook.hpp"
#include <vector>
#include <set>

namespace axis { namespace application { namespace output { namespace workbooks {

class ResultWorkbook::Pimpl
{
public:
  Pimpl(void);
  ~Pimpl(void);

  bool NodeRecordsetExists(const axis::String& nodeSetName) const;
  bool ElementRecordsetExists(const axis::String& elementSetName) const;
  bool RecordsetExists(const axis::application::output::recordsets::ResultRecordset& recordset) const;
  axis::application::output::recordsets::ResultRecordset& GetNodeRecordset(const axis::String& nodeSetName);
  axis::application::output::recordsets::ResultRecordset& GetElementRecordset(const axis::String& elementSetName);

  typedef std::vector<std::pair<axis::String, axis::application::output::recordsets::ResultRecordset *>> 
          RecordsetList;
  typedef std::set<axis::application::output::recordsets::ResultRecordset *> RecordsetSet;

  RecordsetList Nodes;
  RecordsetList Elements;
  RecordsetSet AllRecordsets;
  axis::application::output::recordsets::ResultRecordset *GenericRecordset;
  axis::application::output::recordsets::ResultRecordset *MainRecordset;
  bool IsOpen;
  bool IsAppend;
  axis::String OutputName;
};

} } } } // namespace axis::application::output::workbooks
