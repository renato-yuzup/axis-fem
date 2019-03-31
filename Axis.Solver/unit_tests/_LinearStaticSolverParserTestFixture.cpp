// #include "LinearStaticSolverParserTestFixture.hpp"
// #include "Services/Language/Syntax/Evaluation/IdValue.hpp"
// #include "Application/Factories/Algorithms/LinearStaticSolverFactory.hpp"
// #include "Application/Factories/Impl/BasicSolverParserProvider.hpp"
// #include "Services/Language/Iterators/InputIterator.hpp"
// #include "Services/Language/Factories/IteratorFactory.hpp"
// #include "Services/Language/Parsing/ParseResult.hpp"
// #include "Domain/Analyses/NumericalModel.hpp"
// #include "Application/Analyses/LocalWorkspace.hpp"
// 
// using namespace Axis::foundation;
// using namespace Axis::Services::Language::Factories;
// using namespace Axis::Services::Language::Iterators;
// using namespace Axis::Services::Language::Parsing;
// using namespace Axis::Application::Parsing::Parsers::Base;
// using namespace Axis::Application::Parsing::Parsers::Impl;
// using namespace Axis::Application::Factories::Base;
// using namespace Axis::Application::Factories::Impl;
// using namespace Axis::Application::Factories::Algorithms;
// using namespace Axis::Services::Language::Syntax::Evaluation;
// using namespace Axis::Domain::Analyses;
// using namespace Axis::Application::Analyses;
// 
// Axis::Application::Factories::Base::SolverParserProvider& LinearStaticSolverParserTestFixture::BuildProvider( void ) const
// {
// 	LinearStaticSolverFactory *factory = new LinearStaticSolverFactory();
// 	BasicSolverParserProvider *provider = new BasicSolverParserProvider();
// 	provider->RegisterFactory(*factory);
// 	return *provider;
// }
// 
// void LinearStaticSolverParserTestFixture::TestConstructor( void )
// {
// 	// check if we can build the provider successfully
// 	try
// 	{
// 		SolverParserProvider& provider = BuildProvider();
// 		delete &provider;
// 	}
// 	catch (...)
// 	{
// 		CFIX_FAIL(_T("Construction failed!"));
// 	}
// }
// 
// void LinearStaticSolverParserTestFixture::TestCanParse( void )
// {
// 	SolverParserProvider& provider = BuildProvider();
// 	ParameterList& params = ParameterList::Create();
// 	params.AddParameter(_T("ANALYSIS_TYPE"), *new IdValue(_T("LINEAR_STATIC")));
// 
// 	// first, check if the provider can accept the RUN_SETTINGS block and the
// 	// specified analysis type
// 	CFIXCC_ASSERT_EQUALS(true, provider.CanParse(_T("RUN_SETTINGS"), params));
// 
// 	// should refuse wrong block name
// 	CFIXCC_ASSERT_EQUALS(false, provider.CanParse(_T("BOGUS_BLOCK"), params));
// 
// 	// should refuse empty parameters
// 	CFIXCC_ASSERT_EQUALS(false, provider.CanParse(_T("RUN_SETTINGS"), ParameterList::Empty));
// 
// 	// should refuse parameters in excess
// 	params.AddParameter(_T("BOGUS_PARAM"), *new IdValue(_T("BOGUS_VALUE")));
// 	CFIXCC_ASSERT_EQUALS(false, provider.CanParse(_T("RUN_SETTINGS"), params));
// 
// 	params.Destroy();
// 	delete &provider;
// }
// 
// void LinearStaticSolverParserTestFixture::TestBuildParser( void )
// {
// 	SolverParserProvider& provider = BuildProvider();
// 	ParameterList& params = ParameterList::Create();
// 	params.AddParameter(_T("ANALYSIS_TYPE"), *new IdValue(_T("LINEAR_STATIC")));
// 
// 	// try to build the parser
// 	try
// 	{
// 		BlockParser& parser = provider.BuildParser(_T("RUN_SETTINGS"), params);
// 		delete &parser;
// 	}
// 	catch (...)
// 	{
// 		params.Destroy();
// 		delete &provider;
// 		CFIX_FAIL(_T("Parser build failed!"));
// 	}
// 
// 	params.Destroy();
// 	delete &provider;
// }
// 
// void LinearStaticSolverParserTestFixture::TestParseFail( void )
// {
// 	SolverParserProvider& provider = BuildProvider();
// 	ParameterList& params = ParameterList::Create();
// 	params.AddParameter(_T("ANALYSIS_TYPE"), *new IdValue(_T("LINEAR_STATIC")));
// 
// 	BlockParser& parser = provider.BuildParser(_T("RUN_SETTINGS"), params);
// 
// 	InputParseContext context;
// 	NumericalModel& analysis = NumericalModel::Create();
// 	Workspace& ws = LocalWorkspace::Create(_T("."));
// 	ws.SetAnalysis(&analysis);
// 
// 	// init parser
// 	parser.SetCurrentWorkspace(ws);
// 	parser.StartContext(context);
// 
// 	// should not accept any declaration outside a inner block
// 	String s = _T("BOGUS DECLARATION");
// 	InputIterator begin = IteratorFactory::CreateStringIterator(s.begin());
// 	InputIterator end = IteratorFactory::CreateStringIterator(s.begin());
// 	ParseResult result = parser.Parse(begin, end);
// 	CFIXCC_ASSERT_EQUALS(ParseResult::FailedMatch, result.GetResult());
// 	CFIXCC_ASSERT_EQUALS(end, result.GetLastReadPosition());
// 
// 	// destroy parser
// 	parser.CloseContext();
// 	params.Destroy();
// 	delete &analysis;
// 	delete &provider;
// }
// 
// void LinearStaticSolverParserTestFixture::TestParseSuccess( void )
// {
// 	SolverParserProvider& provider = BuildProvider();
// 	ParameterList& params = ParameterList::Create();
// 	params.AddParameter(_T("ANALYSIS_TYPE"), *new IdValue(_T("LINEAR_STATIC")));
// 
// 	BlockParser& parser = provider.BuildParser(_T("RUN_SETTINGS"), params);
// 
// 	InputParseContext context;
// 	NumericalModel& analysis = NumericalModel::Create();
// 	Workspace& analysis = LocalWorkspace::Create(_T("."));
// 	analysis.SetAnalysis(&analysis);
// 
// 	// init parser
// 	parser.SetCurrentWorkspace(analysis);
// 	parser.StartContext(context);
// 
// 	// destroy parser
// 	parser.CloseContext();
// 	params.Destroy();
// 	analysis.Destroy();
// 	delete &analysis;
// 	delete &provider;
// }