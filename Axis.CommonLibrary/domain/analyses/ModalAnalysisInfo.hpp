#pragma once
#include "AnalysisInfo.hpp"
#include "foundation/Axis.CommonLibrary.hpp"

namespace axis { namespace domain { namespace analyses {

/**
 * @brief Carries information and state of a modal analysis.
**/
class AXISCOMMONLIBRARY_API ModalAnalysisInfo : public AnalysisInfo
{
public:

  /**
   * @brief Constructor.
   *
   * @param firstModeIndex  Index of the first mode.
   * @param lastModeIndex   Index of the last mode.
  **/
  ModalAnalysisInfo(long firstModeIndex, long lastModeIndex);
  virtual ~ModalAnalysisInfo(void);
  virtual void Destroy(void) const;

  /**
   * @brief Return the analysis type this object describes.
   *
   * @return  The analysis type.
  **/
  virtual AnalysisType GetAnalysisType( void ) const;

  /**
   * @brief Returns the last calculated frequency.
   *
   * @return  The last calculated frequency.
  **/
  real GetCurrentFrequency(void) const;

  /**
   * @brief Sets the last calculated frequency.
   *
   * @param freq  The last calculated frequency.
  **/
  void SetCurrentFrequency(real freq);

  /**
   * @brief Returns the index of the last calculated mode.
   *
   * @return  The index of the last calculated mode.
  **/
  long GetCurrentModeIndex(void) const;

  /**
   * @brief Sets the index of the last calculated mode.
   *
   * @param index The index of the last calculated mode.
  **/
  void SetCurrentModeIndex(long index);

  /**
   * @brief Returns the index of the first mode to be calculated.
   *
   * @return  The first mode index.
  **/
  long GetFirstModeIndex(void) const;

  /**
   * @brief Returns the index of the last mode to be calculated.
   *
   * @return  The last mode index.
  **/
  long GetLastModeIndex(void) const;

  virtual AnalysisInfo& Clone( void ) const;
private:
  long firstMode_, lastMode_, currentMode_;
  real currentFrequency_;
};

} } } // namespace axis::domain::analyses
