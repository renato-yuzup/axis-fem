#pragma once
#if defined DEBUG || defined _DEBUG

#include "unit_tests.hpp"
#include "application/factories/collectors/GeneralNodeCollectorParser.hpp"
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

TEST_CLASS(StandardNodeCollectorParserTest)
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

  TEST_METHOD(TestParseDisplacement)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL DISPLACEMENT X, Y, Z ON SET bogus SCALE = 10");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    aafc::CollectorParseResult result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kDisplacement, result.GetCollectorType());
    Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 10) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MAXIMUM NODAL DISPLACEMENT X, Z ON SET bogus");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kDisplacement, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MINIMUM NODAL DISPLACEMENT Z");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kDisplacement, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD NODAL DISPLACEMENT");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kDisplacement, result.GetCollectorType());
    Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));
  }

  TEST_METHOD(TestParseAcceleration)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL ACCELERATION Z, Y, X ON SET test SCALE = -4.5");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    aafc::CollectorParseResult result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kAcceleration, result.GetCollectorType());
    Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() + 4.5) < 1e-5);
    Assert::AreEqual(_T("test"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MAXIMUM NODAL ACCELERATION X, Z ON SET bogus");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kAcceleration, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MINIMUM NODAL ACCELERATION Y");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kAcceleration, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(false, result.ShouldCollectDirection(2));

    s = _T("RECORD NODAL ACCELERATION");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kAcceleration, result.GetCollectorType());
    Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));
  }

  TEST_METHOD(TestParseVelocity)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL VELOCITY ALL ON SET bogus SCALE = 3.5");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    aafc::CollectorParseResult result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kVelocity, result.GetCollectorType());
    Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 3.5) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MAXIMUM NODAL VELOCITY Y, Z ON SET test");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kVelocity, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T("test"), result.GetTargetSetName());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MINIMUM NODAL VELOCITY Z");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kVelocity, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD NODAL VELOCITY");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kVelocity, result.GetCollectorType());
    Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));
  }

  TEST_METHOD(TestParseExternalLoads)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL LOAD X,Y,Z ON SET bogus SCALE = 2");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    aafc::CollectorParseResult result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kExternalLoad, result.GetCollectorType());
    Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 2) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MAXIMUM NODAL LOAD X, Z ON SET bogus");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kExternalLoad, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MINIMUM NODAL LOAD Z");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kExternalLoad, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD NODAL LOAD");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kExternalLoad, result.GetCollectorType());
    Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));
  }

  TEST_METHOD(TestParseReactionForces)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL REACTION X, Y, Z ON SET test SCALE = 5");
    asli::InputIterator begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    asli::InputIterator end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    aafc::CollectorParseResult result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kReactionForce, result.GetCollectorType());
    Assert::AreEqual(aaocs::kAverage, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 5) < 1e-5);
    Assert::AreEqual(_T("test"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MAXIMUM NODAL REACTION X, Z ON SET bogus");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kReactionForce, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMaximum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T("bogus"), result.GetTargetSetName());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD MINIMUM NODAL REACTION Z");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kReactionForce, result.GetCollectorType());
    Assert::AreEqual(aaocs::kMinimum, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(false, result.ShouldCollectDirection(0));
    Assert::AreEqual(false, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));

    s = _T("RECORD NODAL REACTION");
    begin = aslf::IteratorFactory::CreateStringIterator(s.begin());
    end   = aslf::IteratorFactory::CreateStringIterator(s.end());
    result = parser.Parse(begin, end);
    Assert::AreEqual(true, result.GetParseResult().IsMatch());
    Assert::AreEqual(end, result.GetParseResult().GetLastReadPosition());
    Assert::AreEqual(aafc::kReactionForce, result.GetCollectorType());
    Assert::AreEqual(aaocs::kNone, result.GetGroupingType());
    Assert::IsTrue(abs(result.GetScaleFactor() - 1.0) < 1e-5);
    Assert::AreEqual(_T(""), result.GetTargetSetName());
    Assert::AreEqual(true, result.DoesActOnWholeModel());
    Assert::AreEqual(true, result.ShouldCollectDirection(0));
    Assert::AreEqual(true, result.ShouldCollectDirection(1));
    Assert::AreEqual(true, result.ShouldCollectDirection(2));
  }

  TEST_METHOD(TestParseStress)
  {
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL STRESS XX, XZ, YZ, YY, ZZ, XY ON SET bogus SCALE = 10");
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

    s = _T("RECORD MAXIMUM NODAL STRESS XX, XZ ON SET bogus");
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

    s = _T("RECORD MINIMUM NODAL STRESS YZ");
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

    s = _T("RECORD NODAL STRESS");
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
    aafc::GeneralNodeCollectorParser parser;
    axis::String s = _T("RECORD AVERAGE NODAL STRAIN XX, XZ, YZ, YY, ZZ, XY ON SET bogus SCALE = 10");
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

    s = _T("RECORD MAXIMUM NODAL STRAIN XX, XZ ON SET bogus");
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

    s = _T("RECORD MINIMUM NODAL STRAIN YZ");
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

    s = _T("RECORD NODAL STRAIN");
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
