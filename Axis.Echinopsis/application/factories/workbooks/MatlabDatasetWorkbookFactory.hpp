#pragma once
#include "application/factories/workbooks/WorkbookFactory.hpp"

namespace axis { namespace application { namespace factories { namespace workbooks {

class MatlabDatasetWorkbookFactory : public WorkbookFactory
{
public:
  MatlabDatasetWorkbookFactory(void);
  ~MatlabDatasetWorkbookFactory(void);
  virtual void Destroy( void ) const;
  virtual bool CanBuild( const axis::String& formatName, 
      const axis::services::language::syntax::evaluation::ParameterList& formatArguments ) const;
  virtual axis::application::output::workbooks::ResultWorkbook& BuildWorkbook( 
      const axis::String& formatName, 
      const axis::services::language::syntax::evaluation::ParameterList& formatArguments );
};

} } } } // namespace axis::application::factories::workbooks
