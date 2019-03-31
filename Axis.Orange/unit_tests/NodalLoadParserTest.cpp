#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"

#include "System.hpp"
#include "AxisString.hpp"
#include "MockSolver.hpp"
#include "MockClockwork.hpp"
#include "MockResultBucket.hpp"
#include "MockParseContext.hpp"
#include "application/parsing/core/ParseContext.hpp"
#include "application/factories/parsers/NodalLoadParserProvider.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/parsing/parsers/BlockParser.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/SymbolTable.hpp"
#include "domain/analyses/NumericalModel.hpp"
#include "domain/collections/NodeSet.hpp"
#include "domain/curves/MultiLineCurve.hpp"
#include "domain/elements/Node.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/syntax/evaluation/NullValue.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "foundation/memory/pointer.hpp"

namespace aao = axis::application::output;
namespace af = axis::foundation;
namespace aapps = axis::application::parsing::parsers;
namespace aaj = axis::application::jobs;
namespace aafp = axis::application::factories::parsers;
namespace aapc = axis::application::parsing::core;
namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace adc = axis::domain::collections;
namespace ade = axis::domain::elements;
namespace adcv = axis::domain::curves;
namespace aslp = axis::services::language::parsing;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aslse = axis::services::language::syntax::evaluation;
namespace afm = axis::foundation::memory;

// These specializations are required for the use of equality asserts in 
// the MS Unit Test Framework.
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const aslp::ParseResult::Result& v)
{
	return axis::String::int_parse((long)v).data();
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const asli::InputIterator& it)
{
	return std::wstring(1, *it);
}



namespace axis { namespace unit_tests { namespace orange {


/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
TEST_CLASS(NodalLoadParserTest)
{
public:
  TEST_METHOD_INITIALIZE(SetUp)
  {
    axis::System::Initialize();
  }

  TEST_METHOD_CLEANUP(TearDown)
  {
    axis::System::Finalize();
  }

	TEST_METHOD(TestProviderCanParse)
	{
		aafp::NodalLoadParserProvider provider;
		aslse::ParameterList& paramList = aslse::ParameterList::Create();

		// this one should be ok
		Assert::AreEqual(true, provider.CanParse(_T("NODAL_LOADS"), paramList));

		// Now, it must refuse all the following sentences:
		// 1) Wrong block name
		// 2) Not empty parameter list
		Assert::AreEqual(false, provider.CanParse(_T("BOGUS_BLOCK_NAME"), paramList));
		paramList.AddParameter(_T("BOGUS_PARAM"), *new aslse::NullValue());
		Assert::AreEqual(false, provider.CanParse(_T("NODAL_LOADS"), paramList));
	}

	TEST_METHOD(TestProviderBuildParser)
	{
		aafp::NodalLoadParserProvider provider;

		try
		{
			// try to build parser
			aapps::BlockParser& parser = provider.BuildParser(_T("NODAL_LOADS"), aslse::ParameterList::Empty);
			delete &parser;
		}
		catch (...)
		{
			Assert::Fail(_T("Failed to create nodal load parser."));
		}
	}

	TEST_METHOD(TestParserParseFail)
	{
		aafp::NodalLoadParserProvider provider;
		aapps::BlockParser& parser = provider.BuildParser(_T("NODAL_LOADS"), aslse::ParameterList::Empty);
		ada::NumericalModel& analysis = ada::NumericalModel::Create();
		MockParseContext context;
		aaj::StructuralAnalysis& ws = *new aaj::StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		parser.SetAnalysis(ws);
		parser.StartContext(context);

		axis::String s= _T("ONSET my_set_id BEHAVES AS my_curve_id ON X DIRECTION"); // note missing space
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s);
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);

		// must fail
		Assert::AreEqual(aslp::ParseResult::FailedMatch, result.GetResult());
		Assert::AreEqual(1, (int)context.EventSummary().GetTotalEventCount());

		parser.CloseContext();
		ws.Destroy();
		delete &parser;
	}

	TEST_METHOD(TestParserParseCrossRef)
	{
		aafp::NodalLoadParserProvider provider;
		aapps::BlockParser& parser = provider.BuildParser(_T("NODAL_LOADS"), aslse::ParameterList::Empty);
		ada::NumericalModel& analysis = ada::NumericalModel::Create();
		MockParseContext context;
    aapc::SymbolTable& st = context.Symbols();
		aaj::StructuralAnalysis& ws = *new aaj::StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		parser.SetAnalysis(ws);
		parser.StartContext(context);

		axis::String s = _T("ON SET my_set_id BEHAVES AS my_curve_id ON X DIRECTION");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s);
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);

		// parse ok
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());

		// should have added cross-ref for the node-set and curve
		Assert::AreEqual(2, (int)st.GetRoundUnresolvedReferenceCount());

		parser.CloseContext();
		ws.Destroy();
		delete &parser;
	}

	TEST_METHOD(TestParserParseSuccess)
	{
		aafp::NodalLoadParserProvider provider;
		aapps::BlockParser& parser = provider.BuildParser(_T("NODAL_LOADS"), aslse::ParameterList::Empty);
		ada::NumericalModel& analysis = ada::NumericalModel::Create();
		MockParseContext context;
    aapc::SymbolTable& st = context.Symbols();
		aaj::StructuralAnalysis& ws = *new aaj::StructuralAnalysis(_T("."));
		ws.SetNumericalModel(analysis);

		// create analysis step
		aaj::AnalysisStep& step = aaj::AnalysisStep::Create(0, 1, *new MockSolver(*new MockClockwork(1)),
                                                        *new MockResultBucket());
		ws.AddStep(step);
    context.SetStepOnFocusIndex(0);
    context.SetStepOnFocus(&ws.GetStep(0));

		// add some information to our analysis object
		adc::NodeSet& nodeSet = *new adc::NodeSet();
		analysis.AddNodeSet(_T("my_set_id"), nodeSet);
    afm::RelativePointer ptr1 = ade::Node::Create(0, 0, 0.0, 0.0, 0.0);
		ade::Node *node1 = absptr<ade::Node>(ptr1);
		node1->InitDofs(3, 0);
		nodeSet.Add(ptr1);
    afm::RelativePointer ptr2 = ade::Node::Create(1, 1, 1.0, 2.0, 3.0);
		ade::Node *node2 = absptr<ade::Node>(ptr2);
		nodeSet.Add(ptr2);
		node2->InitDofs(3, 3);
		afm::RelativePointer curvePtr = adcv::MultiLineCurve::Create(1);
		analysis.Curves().Add(_T("my_curve_id"), curvePtr);

		// indicate that we have initialized dof's
    st.DefineOrRefreshSymbol(_T("0"), aapc::SymbolTable::kNodeDof);
    st.DefineOrRefreshSymbol(_T("1"), aapc::SymbolTable::kNodeDof);

		// init parser
		parser.SetAnalysis(ws);
		parser.StartContext(context);

		axis::String s = _T("ON SET my_set_id BEHAVES AS my_curve_id ON X,Z DIRECTION");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s);
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);

		// parse ok
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(0, (int)context.EventSummary().GetTotalEventCount());

		// should not have added cross-refs
		Assert::AreEqual(0, (int)st.GetRoundUnresolvedReferenceCount());

		// check if we have nodal loads applied to the specified nodes and dof's
		Assert::AreEqual(true,  step.NodalLoads().Contains(node1->GetDoF(0)));
		Assert::AreEqual(false, step.NodalLoads().Contains(node1->GetDoF(1)));
		Assert::AreEqual(true,  step.NodalLoads().Contains(node1->GetDoF(2)));
		Assert::AreEqual(true,  step.NodalLoads().Contains(node2->GetDoF(0)));
		Assert::AreEqual(false, step.NodalLoads().Contains(node2->GetDoF(1)));
		Assert::AreEqual(true,  step.NodalLoads().Contains(node2->GetDoF(2)));

		// assert nodal loads symbols were created correctly
    Assert::AreEqual(true, st.IsSymbolDefined(_T("0@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("0@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("1@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("1@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition));

		parser.CloseContext();
		delete &parser;
		ws.Destroy();
	}

};

} } }

#endif


