#if defined DEBUG || defined _DEBUG

#include "MockWorkbookFactory.hpp"
#include "MockWorkbook.hpp"

axis::unit_tests::physalis::MockWorkbookFactory::~MockWorkbookFactory( void )
{
  // nothing to do here
}

void axis::unit_tests::physalis::MockWorkbookFactory::Destroy( void ) const
{
  delete this;
}

bool axis::unit_tests::physalis::MockWorkbookFactory::CanBuild( const axis::String& formatName, const axis::services::language::syntax::evaluation::ParameterList& formatArguments ) const
{
  return _T("TEST_FORMAT") == formatName && formatArguments.IsEmpty();
}

axis::application::output::workbooks::ResultWorkbook& axis::unit_tests::physalis::MockWorkbookFactory::BuildWorkbook( const axis::String& formatName, const axis::services::language::syntax::evaluation::ParameterList& formatArguments )
{
  return *new MockWorkbook();
}

#endif