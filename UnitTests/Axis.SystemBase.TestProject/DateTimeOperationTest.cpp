#include "stdafx.h"
#include "CppUnitTest.h"
#include "foundation/date_time/Timestamp.hpp"
#include "AxisString.hpp"
#include "System.hpp"
#include "foundation/memory/HeapStackArena.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

using namespace axis::foundation::date_time;
using namespace axis;

namespace AxisSystemBaseTestProject
{
	TEST_CLASS(DateTimeOperationTest)
	{
	public:
		TEST_METHOD_INITIALIZE(SetUp)
    {
      // reboot system memory if needed
      if (axis::System::IsSystemReady())
      {
        axis::System::Finalize();
      }
      axis:System::Initialize();
    }

    TEST_METHOD_CLEANUP(TearDown)
    {
      axis::System::Finalize();
    }

		TEST_METHOD(TestLocalTime)
		{
			Timestamp today = Timestamp::GetLocalTime();

			Assert::AreEqual(true, today.IsLocalTime());
			Assert::AreEqual(false, today.IsUTCTime());

			String localTimeStr = _T("Today is ");

			localTimeStr += String::int_parse((long)today.GetDate().GetYear()) + _T("-");
			localTimeStr += String::int_parse((long)today.GetDate().GetMonth(), 2).replace(_T(" "), _T("0")) + _T("-");
			localTimeStr += String::int_parse((long)today.GetDate().GetDay(), 2).replace(_T(" "), _T("0")) + _T(" ");

			localTimeStr += String::int_parse((long)today.GetTime().GetHours()) + _T(":");
			localTimeStr += String::int_parse((long)today.GetTime().GetMinutes(), 2).replace(_T(" "), _T("0")) + _T(":");
			localTimeStr += String::int_parse((long)today.GetTime().GetSeconds(), 2).replace(_T(" "), _T("0")) + _T(".");
			localTimeStr += String::int_parse((long)today.GetTime().GetMilliseconds(), 3).replace(_T(" "), _T("0"));

			localTimeStr += _T(" (local time)");

			Logger::WriteMessage(localTimeStr.c_str());
		}

		TEST_METHOD(TestUTCTime)
		{
			Timestamp today = Timestamp::GetUTCTime();

			Assert::AreEqual(false, today.IsLocalTime());
			Assert::AreEqual(true, today.IsUTCTime());

			String utcTimeStr = _T("Today is ");

			utcTimeStr += String::int_parse((long)today.GetDate().GetYear()) + _T("-");
			utcTimeStr += String::int_parse((long)today.GetDate().GetMonth(), 2).replace(_T(" "), _T("0")) + _T("-");
			utcTimeStr += String::int_parse((long)today.GetDate().GetDay(), 2).replace(_T(" "), _T("0")) + _T(" ");

			utcTimeStr += String::int_parse((long)today.GetTime().GetHours()) + _T(":");
			utcTimeStr += String::int_parse((long)today.GetTime().GetMinutes(), 2).replace(_T(" "), _T("0")) + _T(":");
			utcTimeStr += String::int_parse((long)today.GetTime().GetSeconds(), 2).replace(_T(" "), _T("0")) + _T(".");
			utcTimeStr += String::int_parse((long)today.GetTime().GetMilliseconds(), 3).replace(_T(" "), _T("0"));

			utcTimeStr += _T(" (Universal Coordinate Time)");

			Logger::WriteMessage(utcTimeStr.c_str());
		}

		TEST_METHOD(TestDateOperation)
		{
			Date date(2010, 5, 3);
			Date date2(2010, 6, 24);
			
			Timespan dateDiff = date2 - date;
			Assert::AreEqual(52L, dateDiff.GetDays());
			Assert::AreEqual(0L, dateDiff.GetHours());
			Assert::AreEqual(0L, dateDiff.GetMinutes());
			Assert::AreEqual(0L, dateDiff.GetSeconds());
			Assert::AreEqual(0L, dateDiff.GetMilliseconds());

			Date newDate = date + dateDiff;
			Assert::AreEqual(date2.GetYear(), newDate.GetYear());
			Assert::AreEqual(date2.GetMonth(), newDate.GetMonth());
			Assert::AreEqual(date2.GetDay(), newDate.GetDay());
		}
		TEST_METHOD(TestTimeOperation)
		{
			Time time(2, 53, 47);			// 02:53:47
			Time time2(5, 8, 33, 450);		// 05:08:33.450

			Timespan timeDiff = time2 - time;

			Assert::AreEqual(0L, timeDiff.GetDays());
			Assert::AreEqual(2L, timeDiff.GetHours());
			Assert::AreEqual(14L, timeDiff.GetMinutes());
			Assert::AreEqual(46L, timeDiff.GetSeconds());
			Assert::AreEqual(450L, timeDiff.GetMilliseconds());
			Assert::AreEqual(134L, timeDiff.GetTotalMinutes());
			Assert::AreEqual(8086L, timeDiff.GetTotalSeconds());

			Time newTime = time + timeDiff;

			Assert::AreEqual(time2.GetHours(), newTime.GetHours());
			Assert::AreEqual(time2.GetMinutes(), newTime.GetMinutes());
			Assert::AreEqual(time2.GetSeconds(), newTime.GetSeconds());
			Assert::AreEqual(time2.GetMilliseconds(), newTime.GetMilliseconds());
		}
		TEST_METHOD(TestTimestampOperation)
		{
			Date d1(2012, 6, 4), d2(2011, 11, 25);	// note that d1 is leap year!
			Time t1(2, 5, 7, 830), t2(8, 0, 9, 753);
			Timestamp ts1(d1, t1), ts2(d2, t2);

			Timespan tsDiff = ts1 - ts2;

			Assert::AreEqual(191L, tsDiff.GetDays());
			Assert::AreEqual(18L, tsDiff.GetHours());
			Assert::AreEqual(4L, tsDiff.GetMinutes());
			Assert::AreEqual(58L, tsDiff.GetSeconds());
			Assert::AreEqual(77L, tsDiff.GetMilliseconds());

			Timestamp newTs = ts2 + tsDiff;

			Assert::AreEqual(ts1.GetDate().GetYear(), newTs.GetDate().GetYear());
			Assert::AreEqual(ts1.GetDate().GetMonth(), newTs.GetDate().GetMonth());
			Assert::AreEqual(ts1.GetDate().GetDay(), newTs.GetDate().GetDay());
			Assert::AreEqual(ts1.GetTime().GetHours(), newTs.GetTime().GetHours());
			Assert::AreEqual(ts1.GetTime().GetMinutes(), newTs.GetTime().GetMinutes());
			Assert::AreEqual(ts1.GetTime().GetSeconds(), newTs.GetTime().GetSeconds());
			Assert::AreEqual(ts1.GetTime().GetMilliseconds(), newTs.GetTime().GetMilliseconds());
		}
	};
}