#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/domain/physics/InfinitesimalState.hpp"

namespace axis { namespace yuzu { namespace domain { namespace integration {

/**********************************************************************************************//**
	* <summary> Defines an point in a space which will be used to
	* 			 evaluate a polynomial in a weighted sum.</summary>
	**************************************************************************************************/
class IntegrationPoint
{
public:
  /**
   * Destructor.
  **/
	GPU_ONLY ~IntegrationPoint(void);

  /**
   * Returns the set of variables that define the physical state of this point.
   *
   * @return The set of physical variables.
  **/
	GPU_ONLY axis::yuzu::domain::physics::InfinitesimalState& State(void);

  /**
   * Returns the set of variables that define the physical state of this point.
   *
   * @return The set of physical variables.
  **/
	GPU_ONLY const axis::yuzu::domain::physics::InfinitesimalState& State(void) const;

  /**
   * The weight of this integration point.
   *
   * @return The weight.
  **/
	GPU_ONLY real& Weight(void);

  /**
   * The weight of this integration point.
   *
   * @return The weight.
  **/
	GPU_ONLY real Weight(void) const;

  /**
   * Returns the x coordinate of this point.
   *
   * @return The x coordinate.
  **/
  GPU_ONLY real& X(void);

  /**
   * Returns the x coordinate of this point.
   *
   * @return The x coordinate.
  **/
  GPU_ONLY real X(void) const;

  /**
   * Returns the y coordinate of this point.
   *
   * @return The y coordinate.
  **/
  GPU_ONLY real& Y(void);

  /**
   * Returns the y coordinate of this point.
   *
   * @return The y coordinate.
  **/
  GPU_ONLY real Y(void) const;

  /**
   * Returns the z coordinate of this point.
   *
   * @return The z coordinate.
  **/
  GPU_ONLY real& Z(void);

  /**
   * Returns the z coordinate of this point.
   *
   * @return The z coordinate.
  **/
  GPU_ONLY real Z(void) const;
private:
  axis::yuzu::foundation::memory::RelativePointer state_;
  coordtype x_, y_, z_;
  real weight_;

  GPU_ONLY IntegrationPoint(void);
};

} } } } // namespace axis::yuzu::domain::integration
