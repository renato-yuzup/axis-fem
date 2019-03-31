#if defined DEBUG || defined _DEBUG

#include "unit_tests/unit_tests.hpp"
#include "System.hpp"
#include "AxisString.hpp"
#include "services/logging/LogFile.hpp"
#include "application/agents/AnalysisAgent.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "application/jobs/StructuralAnalysis.hpp"
#include "MockClockwork.hpp"
#include "MockSolver.hpp"
#include "MockResultBucket.hpp"
#include "services/messaging/MessageListener.hpp"
#include "application/output/collectors/messages/AnalysisStartupMessage.hpp"
#include "application/output/collectors/messages/AnalysisEndMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepEndMessage.hpp"
#include "application/output/collectors/messages/AnalysisStepStartMessage.hpp"
#include "application/output/ResultBucket.hpp"

namespace aaa = axis::application::agents;
namespace aaj = axis::application::jobs;
namespace aao = axis::application::output;
namespace aaocm = axis::application::output::collectors::messages;
namespace adal = axis::domain::algorithms;
namespace ada = axis::domain::analyses;
namespace af = axis::foundation;
namespace aslg = axis::services::logging;
namespace asmm = axis::services::messaging;
namespace aaocm = axis::application::output::collectors::messages;

namespace axis { namespace unit_tests { namespace core {

/* ================================================================================================================== */
/* ============================================= OUR TEST FIXTURE CLASS ============================================= */
TEST_CLASS(AnalysisAgentTest)
{
private:
	class AnalysisAgentListener : public asmm::MessageListener
	{
	private:
		int _state;
		int _stepCount, _auxStepCount;
	public:
		AnalysisAgentListener(void)
		{
			_state = 0;
			_stepCount = 0; _auxStepCount = 0;
		}

		virtual void DoProcessResultMessage( axis::services::messaging::ResultMessage& volatileMessage ) 
		{
			if (aaocm::AnalysisStartupMessage::IsOfKind(volatileMessage))
			{
				if (_state == 0)
				{
					_state = 1;
				}
				else
				{
					Assert::Fail(_T("Unexpected dispatch of AnalysisStartupMessage."));
				}
			}
			if (aaocm::AnalysisStepStartMessage::IsOfKind(volatileMessage))
			{
				if (_state == 1)
				{
					_state = 2;
					_stepCount++; _auxStepCount++;
				}
				else
				{
					Assert::Fail(_T("Unexpected dispatch of AnalysisStepStartMessage_Old."));
				}
			}
			if (aaocm::AnalysisStepEndMessage::IsOfKind(volatileMessage))
			{
				if (_state == 2)
				{
					_state = 1;
					_auxStepCount--;
				}
				else
				{
					Assert::Fail(_T("Unexpected dispatch of AnalysisStepEndMessage_Old."));
				}
			}
			if (aaocm::AnalysisEndMessage::IsOfKind(volatileMessage))
			{
				if (_state == 1)
				{
					_state = -1;
				}
				else
				{
					Assert::Fail(_T("Unexpected dispatch of AnalysisEndMessage."));
				}
			}
		}

		bool IsStepNestingOk(void) const
		{
			return _auxStepCount == 0;
		}

		int GetStepCount(void) const
		{
			return _stepCount;
		}
	};


	aaj::StructuralAnalysis& CreateTestBenchWorkspace(void) const
	{
		aaj::StructuralAnalysis& analysis = *new aaj::StructuralAnalysis(_T("."));
		ada::NumericalModel& model = ada::NumericalModel::Create();
    aaj::AnalysisStep& step1 = 
              aaj::AnalysisStep::Create(0, 1, *new MockSolver(*new MockClockwork(0.2)),
                                        *new MockResultBucket());
    aaj::AnalysisStep& step2 = 
              aaj::AnalysisStep::Create(1, 2.5, *new MockSolver(*new MockClockwork(0.2)),
                                        *new MockResultBucket());
		analysis.SetNumericalModel(model);
		analysis.AddStep(step1);
		analysis.AddStep(step2);
		return analysis;
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

// 	TEST_METHOD(TestMessageDispatchOrder)
// 	{
// 		aaj::StructuralAnalysis& analysis = CreateTestBenchWorkspace();
// 		AnalysisAgentListener listener;
// 		aaa::AnalysisAgent agent;
// 
// 		agent.ConnectListener(listener);
// 		agent.SetUp(analysis);
// 
// 		agent.Run();
// 
// 		Assert::AreEqual(true, listener.IsStepNestingOk());
// 		Assert::AreEqual(2, listener.GetStepCount());
// 	}
};

} } }


#endif // DEBUG
