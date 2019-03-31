#if defined _DEBUG || defined DEBUG

#include "unit_tests.hpp"
#include "application/factories/parsers/BlockProvider.hpp"
#include <exception>
#include "application/factories/parsers/LinearHexahedronSimpleParserFactory.hpp"
#include "application/factories/parsers/LinearHexahedronFlaBelytschkoParserFactory.hpp"
#include "application/parsing/core/SectionDefinition.hpp"
#include "Services/Language/Syntax/Evaluation/IdValue.hpp"
#include "Services/Language/Syntax/Evaluation/NumberValue.hpp"
#include "System.hpp"

namespace aapc = axis::application::parsing::core;
namespace aafp = axis::application::factories::parsers;
namespace ada = axis::domain::analyses;
namespace aslse = axis::services::language::syntax::evaluation;
namespace aapps = axis::application::parsing::parsers;

namespace axis { namespace unit_tests { namespace standard_elements {

  class MockProvider : public axis::application::factories::parsers::BlockProvider
  {
    virtual bool CanParse( const axis::String& blockName, 
                           const aslse::ParameterList& paramList )
    {
      return false;
    }

    virtual aapps::BlockParser& BuildParser( const axis::String& contextName, 
                                             const aslse::ParameterList& paramList )
    {
      throw std::exception("The method or operation is not implemented.");
    }

    virtual const char * GetFeaturePath( void ) const
    {
      return "";
    }

    virtual const char * GetFeatureName( void ) const
    {
      return "";
    }
  };


TEST_CLASS(ElementParserFactoryTest)
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

  TEST_METHOD(TestLinearHexaFlanaganBelytschkoParserFactoryOkTest)
  {
    MockProvider provider;
    aafp::LinearHexahedronFlaBelytschkoParserFactory factory(provider);
    aapc::SectionDefinition def(_T("LINEAR_HEXAHEDRON"));
    def.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("yes")));

    bool ok = factory.CanBuild(def);
    Assert::AreEqual(true, ok);

    def.AddProperty(_T("STABILIZATION_COEFFICIENT"), aslse::NumberValue(0.7));
    ok = factory.CanBuild(def);
    Assert::AreEqual(true, ok);
  }
  TEST_METHOD(TestLinearHexahedronParserFactoryOkTest)
  {
    MockProvider provider;
    aafp::LinearHexahedronSimpleParserFactory factory(provider);
    aapc::SectionDefinition def(_T("LINEAR_HEXAHEDRON"));
    bool ok = factory.CanBuild(def);
    Assert::AreEqual(true, ok);

    def.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("no")));
    ok = factory.CanBuild(def);
    Assert::AreEqual(true, ok);

    def.AddProperty(_T("INTEGRATION_TYPE"), aslse::IdValue(_T("REDUCED")));
    ok = factory.CanBuild(def);
    Assert::AreEqual(true, ok);
  }
  TEST_METHOD(TestLinearHexaFlanaganBelytschkoParserFactoryFailTest)
  {
    MockProvider provider;
    aafp::LinearHexahedronFlaBelytschkoParserFactory factory(provider);
    aapc::SectionDefinition bogusElemTypeDef(_T("BOGUS_HEXAHEDRON"));
    bool ok = factory.CanBuild(bogusElemTypeDef);
    Assert::AreEqual(false, ok);

    aapc::SectionDefinition bogusDef1(_T("LINEAR_HEXAHEDRON"));
    ok = factory.CanBuild(bogusDef1);
    Assert::AreEqual(false, ok);
    bogusDef1.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("no")));
    ok = factory.CanBuild(bogusDef1);
    Assert::AreEqual(false, ok);

    aapc::SectionDefinition bogusDef2(_T("LINEAR_HEXAHEDRON"));
    bogusDef2.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("yes")));
    bogusDef2.AddProperty(_T("BOGUS_PARAM"), aslse::IdValue(_T("XX")));
    ok = factory.CanBuild(bogusDef2);
    Assert::AreEqual(false, ok);
  }
  TEST_METHOD(TestLinearHexahedronParserFactoryFailTest)
  {
    MockProvider provider;
    aafp::LinearHexahedronSimpleParserFactory factory(provider);
    aapc::SectionDefinition bogusElemTypeDef(_T("BOGUS_HEXAHEDRON"));
    bool ok = factory.CanBuild(bogusElemTypeDef);
    Assert::AreEqual(false, ok);

    aapc::SectionDefinition bogusDef1(_T("LINEAR_HEXAHEDRON"));
    ok = factory.CanBuild(bogusDef1);
    Assert::AreEqual(true, ok);
    bogusDef1.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("yes")));
    ok = factory.CanBuild(bogusDef1);
    Assert::AreEqual(false, ok);

    aapc::SectionDefinition bogusDef2(_T("LINEAR_HEXAHEDRON"));
    bogusDef2.AddProperty(_T("HOURGLASS_CONTROL"), aslse::IdValue(_T("yes")));
    bogusDef2.AddProperty(_T("BOGUS_PARAM"), aslse::IdValue(_T("XX")));
    ok = factory.CanBuild(bogusDef2);
    Assert::AreEqual(false, ok);
  }
};

} } } // namespace axis::unit_tests::standard_elements

#endif
