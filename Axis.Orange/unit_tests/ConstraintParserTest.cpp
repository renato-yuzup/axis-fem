#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include "MockClockwork.hpp"
#include "MockParseContext.hpp"
#include "MockSolver.hpp"
#include "System.hpp"
#include "AxisString.hpp"
#include "MockResultBucket.hpp"
#include "application/factories/boundary_conditions/LockConstraintFactory.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/jobs/AnalysisStep.hpp"
#include "application/locators/ConstraintParserLocator.hpp"
#include "application/output/ResultBucket.hpp"
#include "application/parsing/core/EventStatistic.hpp"
#include "application/parsing/core/SymbolTable.hpp"

#include "domain/collections/NodeSet.hpp"
#include "domain/elements/Node.hpp"

#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "services/language/parsing/ParseResult.hpp"
#include "services/language/syntax/evaluation/ParameterList.hpp"
#include "services/language/syntax/evaluation/NullValue.hpp"

#include "foundation/memory/pointer.hpp"

namespace af = axis::foundation;
namespace aal = axis::application::locators;
namespace aapps = axis::application::parsing::parsers;
namespace aafp = axis::application::factories::parsers;
namespace aafb = axis::application::factories::boundary_conditions;
namespace aapc = axis::application::parsing::core;
namespace aaj = axis::application::jobs;
namespace aafb = axis::application::factories::boundary_conditions;
namespace aao = axis::application::output;
namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace ade = axis::domain::elements;
namespace adc = axis::domain::collections;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aslp = axis::services::language::parsing;
namespace asli = axis::services::language::iterators;
namespace aslf = axis::services::language::factories;
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
TEST_CLASS(ConstraintParserTestFixture)
{
private:
	aal::ConstraintParserLocator& BuildLocator() const
	{
		aal::ConstraintParserLocator& locator = *new aal::ConstraintParserLocator();
		aafb::LockConstraintFactory *builder = new aafb::LockConstraintFactory();
		locator.RegisterFactory(*builder);
		return locator;
	}

	aaj::StructuralAnalysis& CreateTestBenchWorkspace(void) const
	{
		aaj::StructuralAnalysis& ws = *new aaj::StructuralAnalysis(_T("."));
		ada::NumericalModel& analysis = ada::NumericalModel::Create();
		ws.SetNumericalModel(analysis);

		// create testbench step
		aaj::AnalysisStep& step = aaj::AnalysisStep::Create(0, 1, 
                                                        *new MockSolver(*new MockClockwork(1)),
                                                        *new MockResultBucket());
		ws.AddStep(step);

		return ws;
	}
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
		aal::ConstraintParserLocator& locator = BuildLocator();
		aslse::ParameterList& paramList = aslse::ParameterList::Create();

		// check for a well-formed block
		Assert::AreEqual(true, locator.CanParse(_T("CONSTRAINTS"), aslse::ParameterList::Empty));

		// should fail on a bogus block name
		Assert::AreEqual(false, locator.CanParse(_T("BOGUS_BLOCK_NAME"), aslse::ParameterList::Empty));

		// should fail on a non-empty parameter list
		paramList.AddParameter(_T("BOGUS_PARAM"), *new aslse::NullValue());
		Assert::AreEqual(false, locator.CanParse(_T("CONSTRAINTS"), paramList));

		paramList.Destroy();
		delete &locator;
	}

	TEST_METHOD(TestProviderBuildParser)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();

		try
		{
			aapps::BlockParser& parser = locator.BuildParser(_T("CONSTRAINTS"), aslse::ParameterList::Empty);
			delete &parser;
		}
		catch (...)
		{	// something went wrong
			delete &locator;
			Assert::Fail(_T("BuildParser failed!"));
		}
		delete &locator;
	}

	TEST_METHOD(TestProviderRecognizeDeclaration)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();

		// this ok should pass
		axis::String s = _T("LOCK\t\t\t SET my_eset_id IN X,z DIRECTIONS\t\t\t");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = locator.TryParse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// this one should fail
		s = _T("BLOCKMOVEMENT OF my_eset_id ON X,z DIRECTIONS\t\t\t");
		begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
		end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		result = locator.TryParse(begin, end);
		Assert::AreEqual(aslp::ParseResult::FailedMatch, result.GetResult());
		Assert::AreEqual(begin, result.GetLastReadPosition());

		delete &locator;
	}

	TEST_METHOD(TestParserParseFail)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();
		aapps::BlockParser& parser = locator.BuildParser(_T("CONSTRAINTS"), aslse::ParameterList::Empty);
		MockParseContext context;
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();

		parser.SetAnalysis(ws);
    context.SetStepOnFocus(&ws.GetStep(0));
		parser.StartContext(context);

		// this should not pass
		axis::String s = _T("BLOCKMOVEMENT OF my_eset_id ON X,z DIRECTIONS\t\t\t");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(false, result.IsMatch());
		Assert::AreEqual(begin, result.GetLastReadPosition());

		parser.CloseContext();

		delete &parser;
		delete &locator;
		ws.Destroy();
	}

	TEST_METHOD(TestParserParseCrossRefUnresolved)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();
		aapps::BlockParser& parser = locator.BuildParser(_T("CONSTRAINTS"), aslse::ParameterList::Empty);
		MockParseContext context;
    aapc::SymbolTable& st = context.Symbols();
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
    context.SetStepOnFocus(&ws.GetStep(0));
		parser.SetAnalysis(ws);
		parser.StartContext(context);

		// this ok should pass
		axis::String s = _T("LOCK SET my_nset_id IN X,z DIRECTIONS\t\t\t");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// node set doesn't exist, so an unresolved entry should exist
		Assert::AreEqual(true, st.IsSymbolCurrentRoundUnresolved(_T("my_nset_id"), aapc::SymbolTable::kNodeSet));
    Assert::AreEqual(1, (int)st.GetRoundUnresolvedReferenceCount());
    Assert::AreEqual(0, (int)st.GetRoundDefinedReferenceCount());

		parser.CloseContext();

		delete &parser;
		delete &locator;
		ws.Destroy();
	}

	TEST_METHOD(TestParserParseCrossRefOverlapped)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();
		aapps::BlockParser& parser = locator.BuildParser(_T("CONSTRAINTS"), aslse::ParameterList::Empty);
		MockParseContext context;
    aapc::SymbolTable& st = context.Symbols();
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		ada::NumericalModel& model = ws.GetNumericalModel();

		// add some data to the analysis object
		adc::NodeSet& set = *new adc::NodeSet();
		model.AddNodeSet(_T("my_nset_id"), set);
    afm::RelativePointer ptr = ade::Node::Create(1, 1, 3);
		ade::Node *node = absptr<ade::Node>(ptr);
		node->InitDofs(3, 0);
		model.Nodes().Add(ptr);
		set.Add(ptr);
    ptr = ade::Node::Create(2, 2, 3);
		node = absptr<ade::Node>(ptr);
		node->InitDofs(3, 3);
		model.Nodes().Add(ptr);
		set.Add(ptr);
    ptr = ade::Node::Create(3, 3, 3);
		node = absptr<ade::Node>(ptr);
		node->InitDofs(3, 6);
		model.Nodes().Add(ptr);
		set.Add(ptr);

		// These informations might look bogus towards what the
		// analysis object really contains, however we will do as it is
		// to simulate another parser which previously have assigned
		// boundary conditions to some of our nodes (making invalid
		// applying boundary conditions to dof's of all nodes in the
		// set).
    st.DefineOrRefreshSymbol(_T("1@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition);
    st.DefineOrRefreshSymbol(_T("3@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition);
    st.DefineOrRefreshSymbol(_T("1"), aapc::SymbolTable::kNodeDof);
    st.DefineOrRefreshSymbol(_T("2"), aapc::SymbolTable::kNodeDof);
    st.DefineOrRefreshSymbol(_T("3"), aapc::SymbolTable::kNodeDof);

		// start parsing!
		context.SetStepOnFocus(&ws.GetStep(0));
    context.SetStepOnFocusIndex(0);
    parser.SetAnalysis(ws);
		parser.StartContext(context);

		axis::String s = _T("LOCK SET my_nset_id IN X,Z DIRECTIONS\t\t\t");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// we know that we have two cases of overlapping, but the
		// parser only catches the first one in order to prevent a
		// massive logging of errors (or so it should be)
		Assert::AreEqual(1, (int)context.EventSummary().GetErrorCount());

		// symbol table should not have been modified
    Assert::AreEqual(5, (int)st.GetRoundDefinedReferenceCount());
    Assert::AreEqual(0, (int)st.GetRoundUnresolvedReferenceCount());

		parser.CloseContext();

		delete &parser;
		delete &locator;
		ws.Destroy();
	}

	TEST_METHOD(TestParserParseSuccess)
	{
		aal::ConstraintParserLocator& locator = BuildLocator();
		aapps::BlockParser& parser = locator.BuildParser(_T("CONSTRAINTS"), aslse::ParameterList::Empty);
		MockParseContext context;
    aapc::SymbolTable& st = context.Symbols();
		aaj::StructuralAnalysis& ws = CreateTestBenchWorkspace();
		ada::NumericalModel& model = ws.GetNumericalModel();
		aaj::AnalysisStep& step = ws.GetStep(0);

		// add some data to the analysis object
		adc::NodeSet& set = *new adc::NodeSet();
		model.AddNodeSet(_T("my_nset_id"), set);
    afm::RelativePointer ptr = ade::Node::Create(1, 1, 3);
		ade::Node *node1 = absptr<ade::Node>(ptr);
		node1->InitDofs(3, 0);
		model.Nodes().Add(ptr);
		set.Add(ptr);
    ptr = ade::Node::Create(2, 2, 3);
		ade::Node *node2 = absptr<ade::Node>(ptr);
		node2->InitDofs(3, 0);
		model.Nodes().Add(ptr);
		set.Add(ptr);
    ptr = ade::Node::Create(3, 3, 3);
		ade::Node *node3 = absptr<ade::Node>(ptr);
		node3->InitDofs(3, 0);
		model.Nodes().Add(ptr);
		set.Add(ptr);

		// these symbols are necessary to indicate that nodal DoFs has
		// been initialized
    st.DefineOrRefreshSymbol(_T("1"), aapc::SymbolTable::kNodeDof);
    st.DefineOrRefreshSymbol(_T("2"), aapc::SymbolTable::kNodeDof);
    st.DefineOrRefreshSymbol(_T("3"), aapc::SymbolTable::kNodeDof);

		// start parsing!
		context.SetStepOnFocus(&ws.GetStep(0));
    context.SetStepOnFocusIndex(0);
    parser.SetAnalysis(ws);
		parser.StartContext(context);

		axis::String s = _T("LOCK SET my_nset_id IN X,Z DIRECTIONS\t\t\t");
		asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
		asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
		aslp::ParseResult result = parser.Parse(begin, end);
		Assert::AreEqual(true, result.IsMatch());
		Assert::AreEqual(end, result.GetLastReadPosition());

		// check if the correct symbols were created
    Assert::AreEqual(true, st.IsSymbolDefined(_T("1@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("1@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("2@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("2@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("3@@0@0"), aapc::SymbolTable::kNodalBoundaryCondition));
    Assert::AreEqual(true, st.IsSymbolDefined(_T("3@@2@0"), aapc::SymbolTable::kNodalBoundaryCondition));

		// check if the constraints really exist
		Assert::AreEqual(true , step.Locks().Contains(node1->GetDoF(0)));
		Assert::AreEqual(false, step.Locks().Contains(node1->GetDoF(1)));
		Assert::AreEqual(true , step.Locks().Contains(node1->GetDoF(2)));
		Assert::AreEqual(true , step.Locks().Contains(node2->GetDoF(0)));
		Assert::AreEqual(false, step.Locks().Contains(node2->GetDoF(1)));
		Assert::AreEqual(true , step.Locks().Contains(node2->GetDoF(2)));
		Assert::AreEqual(true , step.Locks().Contains(node3->GetDoF(0)));
		Assert::AreEqual(false, step.Locks().Contains(node3->GetDoF(1)));
		Assert::AreEqual(true , step.Locks().Contains(node3->GetDoF(2)));

		parser.CloseContext();

		delete &parser;
		delete &locator;
		ws.Destroy();
	}

};



} } }

#endif

