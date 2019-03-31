#pragma once
#include "AnalysisInfo.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace analyses {

/**
 * @brief Carries information and state of a static analysis.
**/
class AXISCOMMONLIBRARY_API StaticAnalysisInfo : public AnalysisInfo
{
public:
  StaticAnalysisInfo(void);
  virtual ~StaticAnalysisInfo(void);
  virtual void Destroy(void) const;

  /**
   * @brief Return the analysis type this object describes.
   *
   * @return  The analysis type.
  **/
  virtual AnalysisType GetAnalysisType( void ) const;
  virtual AnalysisInfo& Clone( void ) const;
};

} } } // namespace axis::domain::analyses
