#pragma once
#include "AnalysisInfo.hpp"
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/basic_types.hpp"

namespace axis { namespace domain { namespace analyses {

/**
 * @brief Carries information and state of a transient analysis.
**/
class AXISCOMMONLIBRARY_API TransientAnalysisInfo : public AnalysisInfo
{
public:
  TransientAnalysisInfo(real startTime, real endTime);
  virtual ~TransientAnalysisInfo(void);
  virtual void Destroy(void) const;

  virtual AnalysisType GetAnalysisType( void ) const;

  real GetStartTime(void) const;
  real GetEndTime(void) const;
  real GetCurrentAnalysisTime(void) const;
  void SetCurrentAnalysisTime(real newTime);
  real GetLastTimeStep(void) const;
  void SetLastTimeStep(real newTimeStep);
  uint64 GetIterationIndex(void) const;
  void SetIterationIndex(uint64 index);
  virtual AnalysisInfo& Clone( void ) const;
private:
  real startTime_, endTime_, currentTime_;
  real lastTimeStep_;
  uint64 iterationIndex_;
};

} } } // namespace axis::domain::analyses
