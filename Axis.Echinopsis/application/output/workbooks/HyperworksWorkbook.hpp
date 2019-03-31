#pragma once
#include "application/output/workbooks/ResultWorkbook.hpp"
#include <list>
#include "services/io/StreamWriter.hpp"

namespace axis { namespace application { namespace output { namespace workbooks {

class HyperworksWorkbook : public ResultWorkbook
{
public:
  HyperworksWorkbook(void);
  ~HyperworksWorkbook(void);
  virtual void Destroy( void ) const;
  virtual bool SupportsAppendOperation( void ) const;
  virtual axis::String GetFormatIdentifier( void ) const;
  virtual axis::String GetFormatTitle( void ) const;
  virtual axis::String GetShortDescription( void ) const;
  virtual bool SupportsNodeRecordset( void ) const;
  virtual bool SupportsElementRecordset( void ) const;
  virtual bool SupportsGenericRecordset( void ) const;
  virtual bool SupportsMainRecordset( void ) const;
  virtual bool IsReady( void ) const;
private:
  typedef std::list<axis::String> tempfile_list;

  virtual axis::application::output::recordsets::ResultRecordset& 
      DoCreateNodeRecordset( const axis::String& nodeSetName );
  virtual axis::application::output::recordsets::ResultRecordset& 
      DoCreateElementRecordset( const axis::String& elementSetName );
  virtual void DoAfterInit( axis::application::jobs::WorkFolder& workFolder );
  virtual void DoAfterOpen( const axis::application::jobs::AnalysisStepInformation& stepInfo );
  virtual void DoBeforeClose( void );
  virtual void DoAfterClose( void );

  axis::services::io::StreamWriter *writer_;
  axis::application::jobs::WorkFolder *workFolder_;

  tempfile_list tempFiles_;
};

} } } } // namespace axis::application::output::workbooks
