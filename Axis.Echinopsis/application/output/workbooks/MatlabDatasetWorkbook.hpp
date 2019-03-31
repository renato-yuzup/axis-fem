#pragma once
#include "application/output/workbooks/ResultWorkbook.hpp"
#include "application/output/recordsets/MatlabDatasetRecordset.hpp"

namespace axis { namespace application { namespace output { namespace workbooks {

/**
 * A workbook capable to write data according to a MATLAB(R) dataset (.m file).
 *
 * @sa ResultWorkbook
 */
class MatlabDatasetWorkbook : public ResultWorkbook
{
public:
  MatlabDatasetWorkbook(const axis::String& variableName);
  virtual ~MatlabDatasetWorkbook(void);

  virtual void Destroy( void ) const;

  virtual bool SupportsAppendOperation( void ) const;

  virtual axis::String GetFormatIdentifier( void ) const;

  virtual axis::String GetFormatTitle( void ) const;

  virtual axis::String GetShortDescription( void ) const;

  virtual bool SupportsNodeRecordset( void ) const;

  virtual bool SupportsElementRecordset( void ) const;

  virtual bool SupportsGenericRecordset( void ) const;

  virtual bool SupportsMainRecordset( void ) const;
private:
  virtual axis::application::output::recordsets::ResultRecordset& DoCreateGenericRecordset( axis::application::jobs::WorkFolder& workFolder );
  virtual void DoAfterOpen( const axis::application::jobs::AnalysisStepInformation& stepInfo );

  axis::application::output::recordsets::MatlabDatasetRecordset *recordset_;
  axis::String variableName_;
};

} } } } // namespace axis::application::output::workbooks
