// #pragma once
// 
// #include <tchar.h>
// #include <cfixcc.h>
// #include "AxisString.hpp"
// #include "Application/Parsing/Core/InputParseContext.hpp"
// #include "Application/Factories/Base/SolverParserProvider.hpp"
// 
// using namespace Axis::foundation;
// using namespace Axis::Application::Factories::Base;
// using namespace Axis::Application::Parsing::Parsers::Base;
// using namespace Axis::Domain::Materials::Base;
// 
// 
// /* ================================================================================================================== */
// /* ============================================= OUR TEST FIXTURE CLASS ============================================= */
// class LinearStaticSolverParserTestFixture : public cfixcc::TestFixture
// {
// private:
// 	Axis::Application::Factories::Base::SolverParserProvider& BuildProvider(void) const;
// public:
// 	void TestConstructor(void);
// 	void TestCanParse(void);
// 	void TestBuildParser(void);
// 	void TestParseFail(void);
// 	void TestParseSuccess(void);
// };
// 
// CFIXCC_BEGIN_CLASS( LinearStaticSolverParserTestFixture )
// 	CFIXCC_METHOD( TestConstructor )
// 	CFIXCC_METHOD( TestCanParse )
// 	CFIXCC_METHOD( TestBuildParser )
// 	CFIXCC_METHOD( TestParseFail )
// 	CFIXCC_METHOD( TestParseSuccess )
// CFIXCC_END_CLASS()
// 
