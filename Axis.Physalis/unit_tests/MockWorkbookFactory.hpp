#pragma once
#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "application/factories/workbooks/WorkbookFactory.hpp"

namespace axis { namespace unit_tests { namespace physalis {

class MockWorkbookFactory : public axis::application::factories::workbooks::WorkbookFactory
{
public:
  virtual ~MockWorkbookFactory(void);
  virtual void Destroy( void ) const;
  virtual bool CanBuild( const axis::String& formatName, const axis::services::language::syntax::evaluation::ParameterList& formatArguments ) const;
  virtual axis::application::output::workbooks::ResultWorkbook& BuildWorkbook( const axis::String& formatName, const axis::services::language::syntax::evaluation::ParameterList& formatArguments );
};

} } } 

#endif
