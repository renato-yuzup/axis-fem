#pragma once
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace analyses {

/**
 * @brief Carries information and state of the current 
 *        numerical analysis.
**/
class AXISCOMMONLIBRARY_API AnalysisInfo
{
public:

  /**
   * @brief Values that represent each possible analysis type.
  **/
  enum AnalysisType
  {
    StaticAnalysis,
    TransientAnalysis,
    ModalAnalysis
  };

  /**
   * @brief Destructor.
  **/
  virtual ~AnalysisInfo(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * @brief Return the analysis type this object describes.
   *
   * @return  The analysis type.
  **/
  virtual AnalysisType GetAnalysisType(void) const = 0;

  /**
   * Makes a deep copy of this object.
   *
   * @return A copy of this object.
  **/
  virtual AnalysisInfo& Clone(void) const = 0;
};

} } } // namespace axis::domain::analyses
