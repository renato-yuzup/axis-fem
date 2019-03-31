#pragma once
#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "MockWorkbook.hpp"
#include "System.hpp"
#include "foundation/ArgumentException.hpp"

namespace aaor = axis::application::output::recordsets;

namespace axis { namespace unit_tests { namespace physalis {

TEST_CLASS(ResultWorkbookTest)
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

  TEST_METHOD(TestCreateNodeRecordset)
  {
    MockWorkbook w;
    w.SetNextRecordsetIndex(1);
    w.CreateNodeRecordset(_T("x"));
    w.SetNextRecordsetIndex(3);
    w.CreateNodeRecordset(_T("y"));

    MockRecordset& n1 = static_cast<MockRecordset&>(w.GetNodeRecordset(_T("x")));
    MockRecordset& n2 = static_cast<MockRecordset&>(w.GetNodeRecordset(_T("y")));
    Assert::AreEqual(1, n1.GetIndex());
    Assert::AreEqual(3, n2.GetIndex());
    Assert::AreEqual(2, w.GetNodeRecordsetCount());
    Assert::AreEqual(0, w.GetElementRecordsetCount());
    
    try
    {
      aaor::ResultRecordset& r = w.GetNodeRecordset(_T("w"));
      Assert::Fail(_T("Unexpected behavior in GetNodeRecordset() method."));
    }
    catch (axis::foundation::ArgumentException&)
    {
    	// ok, this is expected
    }
    catch (...)
    {
      Assert::Fail(_T("GetNodeRecordset() method thrown an unexpected exception."));
    }
    try
    {
      aaor::ResultRecordset& r = w.GetElementRecordset(_T("x"));
      Assert::Fail(_T("A node recordset was retrieved using GetElementRecordset() method."));
    }
    catch (axis::foundation::ArgumentException&)
    {
      // ok, this is expected
    }
    catch (...)
    {
      Assert::Fail(_T("GetElementRecordset() method thrown an unexpected exception."));
    }

  }
  TEST_METHOD(TestCreateElementRecordset)
  {
    MockWorkbook w;
    w.SetNextRecordsetIndex(2);
    w.CreateElementRecordset(_T("z"));
    w.SetNextRecordsetIndex(4);
    w.CreateElementRecordset(_T("w"));

    MockRecordset& e1 = static_cast<MockRecordset&>(w.GetElementRecordset(_T("z")));
    MockRecordset& e2 = static_cast<MockRecordset&>(w.GetElementRecordset(_T("w")));
    Assert::AreEqual(2, e1.GetIndex());
    Assert::AreEqual(4, e2.GetIndex());
    Assert::AreEqual(0, w.GetNodeRecordsetCount());
    Assert::AreEqual(2, w.GetElementRecordsetCount());

    try
    {
      aaor::ResultRecordset& r = w.GetNodeRecordset(_T("w"));
      Assert::Fail(_T("An element recordset was retrieved using GetNodeRecordset() method."));
    }
    catch (axis::foundation::ArgumentException&)
    {
      // ok, this is expected
    }
    catch (...)
    {
      Assert::Fail(_T("GetNodeRecordset() method thrown an unexpected exception."));
    }
    try
    {
      aaor::ResultRecordset& r = w.GetElementRecordset(_T("x"));
      Assert::Fail(_T("Unexpected behavior in GetElementRecordset() method."));
    }
    catch (axis::foundation::ArgumentException&)
    {
      // ok, this is expected
    }
    catch (...)
    {
      Assert::Fail(_T("GetElementRecordset() method thrown an unexpected exception."));
    }

  }
};

} } }

#endif