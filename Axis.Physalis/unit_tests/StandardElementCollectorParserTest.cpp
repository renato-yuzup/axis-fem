#pragma once
#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "application/factories/collectors/GeneralElementCollectorParser.hpp"
#include "services/language/factories/IteratorFactory.hpp"
#include "services/language/iterators/InputIterator.hpp"
#include "System.hpp"

namespace aafc = axis::application::factories::collectors;
namespace aslf = axis::services::language::factories;
namespace asli = axis::services::language::iterators;
namespace aaocs = axis::application::output::collectors::summarizers;
namespace aslp = axis::services::language::parsing;

template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const asli::InputIterator& q)
{
  return std::wstring(*q + _T(""));
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const aafc::CollectorType& q)
{
  return std::wstring(axis::String::int_parse((int)q).data());
}
template <>
std::wstring Microsoft::VisualStudio::CppUnitTestFramework::ToString(const aaocs::SummaryType& q)
{
  return std::wstring(axis::String::int_parse((int)q).data());
}

namespace axis { namespace unit_tests { namespace physalis {

  TEST_CLASS(StandardElementCollectorParserTest)
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

    TEST_METHOD(TestParseStress)
    {
      aafc::GeneralElementCollectorParser parser;
      axis::String s = _T("RECORD AVERAGE ELEMENT STRESS XX, XZ, YZ, YY, ZZ, XY ON SET bogus SCALE = 10");
      asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      aafc::CollectorParseResult result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStress, result.GetCollectorType());
      Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 10) < 1e-5);
      Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(true, result.ShouldCollectDirection(1));
      Assert::AreEqual(true, result.ShouldCollectDirection(2));
      Assert::AreEqual(true, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));

      s = _T("RECORD MAXIMUM ELEMENT STRESS XX, XZ ON SET bogus");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStress, result.GetCollectorType());
      Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(false, result.ShouldCollectDirection(1));
      Assert::AreEqual(false, result.ShouldCollectDirection(2));
      Assert::AreEqual(false, result.ShouldCollectDirection(3));
      Assert::AreEqual(false, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));

      s = _T("RECORD MINIMUM ELEMENT STRESS YZ");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStress, result.GetCollectorType());
      Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T(""), result.GetTargetSetName());
      Assert::AreEqual(true, result.DoesActOnWholeModel());
      Assert::AreEqual(false, result.ShouldCollectDirection(0));
      Assert::AreEqual(false, result.ShouldCollectDirection(1));
      Assert::AreEqual(false, result.ShouldCollectDirection(2));
      Assert::AreEqual(false, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(false, result.ShouldCollectDirection(5));

      s = _T("RECORD ELEMENT STRESS");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStress, result.GetCollectorType());
      Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T(""), result.GetTargetSetName());
      Assert::AreEqual(true, result.DoesActOnWholeModel());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(true, result.ShouldCollectDirection(1));
      Assert::AreEqual(true, result.ShouldCollectDirection(2));
      Assert::AreEqual(true, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));
    }

    TEST_METHOD(TestParseStrain)
    {
      aafc::GeneralElementCollectorParser parser;
      axis::String s = _T("RECORD AVERAGE ELEMENT STRAIN XX, XZ, YZ, YY, ZZ, XY ON SET bogus SCALE = 10");
      asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      aafc::CollectorParseResult result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStrain, result.GetCollectorType());
      Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 10) < 1e-5);
      Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(true, result.ShouldCollectDirection(1));
      Assert::AreEqual(true, result.ShouldCollectDirection(2));
      Assert::AreEqual(true, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));

      s = _T("RECORD MAXIMUM ELEMENT STRAIN XX, XZ ON SET bogus");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStrain, result.GetCollectorType());
      Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(false, result.ShouldCollectDirection(1));
      Assert::AreEqual(false, result.ShouldCollectDirection(2));
      Assert::AreEqual(false, result.ShouldCollectDirection(3));
      Assert::AreEqual(false, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));

      s = _T("RECORD MINIMUM ELEMENT STRAIN YZ");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStrain, result.GetCollectorType());
      Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T(""), result.GetTargetSetName());
      Assert::AreEqual(true, result.DoesActOnWholeModel());
      Assert::AreEqual(false, result.ShouldCollectDirection(0));
      Assert::AreEqual(false, result.ShouldCollectDirection(1));
      Assert::AreEqual(false, result.ShouldCollectDirection(2));
      Assert::AreEqual(false, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(false, result.ShouldCollectDirection(5));

      s = _T("RECORD ELEMENT STRAIN");
      begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
      end   = aslf::IteratorFactory::CreateStringIterator(s.end());
      result = parser.Parse(begin, end);
      Assert::AreEqual(true, result.GetParseResult().IsMatch());
      Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
      Assert::AreEqual(aafc::kStrain, result.GetCollectorType());
      Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
      Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
      Assert::AreEqual(_T(""), result.GetTargetSetName());
      Assert::AreEqual(true, result.DoesActOnWholeModel());
      Assert::AreEqual(true, result.ShouldCollectDirection(0));
      Assert::AreEqual(true, result.ShouldCollectDirection(1));
      Assert::AreEqual(true, result.ShouldCollectDirection(2));
      Assert::AreEqual(true, result.ShouldCollectDirection(3));
      Assert::AreEqual(true, result.ShouldCollectDirection(4));
      Assert::AreEqual(true, result.ShouldCollectDirection(5));
    }
  };

} } } 

#endif
