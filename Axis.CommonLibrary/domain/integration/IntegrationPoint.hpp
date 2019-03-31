#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/memory/RelativePointer.hpp"

namespace axis { namespace domain { namespace integration {

/**********************************************************************************************//**
	* <summary> Defines an point in a space which will be used to
	* 			 evaluate a polynomial in a weighted sum.</summary>
	**************************************************************************************************/
class AXISCOMMONLIBRARY_API IntegrationPoint
{
public:
  /**
   * Creates a new point in the origin.
  **/
	IntegrationPoint(void);

  /**
   * Creates a new point with null z-coordinate (zero).
   *
   * @param x The x-coordinate of the point.
   * @param y The y-coordinate of the point.
  **/
	IntegrationPoint(coordtype x, coordtype y);

  /**
   * Creates a new point.
   *
   * @param x The x-coordinate of the point.
   * @param y The y-coordinate of the point.
   * @param z The z-coordinate of the point.
  **/
	IntegrationPoint(coordtype x, coordtype y, coordtype z);

  /**
   * Creates a new point.
   *
   * @param x      The x-coordinate of the point.
   * @param y      The y-coordinate of the point.
   * @param z      The z-coordinate of the point.
   * @param weight The point weight.
  **/
	IntegrationPoint(coordtype x, coordtype y, coordtype z, real weight);

  /**
   * Destructor.
  **/
	~IntegrationPoint(void);

  /**
   * Destroys this object.
  **/
	void Destroy(void) const;

  /**
   * Returns the set of variables that define the physical state of this point.
   *
   * @return The set of physical variables.
  **/
	axis::domain::physics::InfinitesimalState& State(void);

  /**
   * Returns the set of variables that define the physical state of this point.
   *
   * @return The set of physical variables.
  **/
	const axis::domain::physics::InfinitesimalState& State(void) const;

  /**
   * The weight of this integration point.
   *
   * @return The weight.
  **/
	real& Weight(void);

  /**
   * The weight of this integration point.
   *
   * @return The weight.
  **/
	real Weight(void) const;

  /**
   * Returns the x coordinate of this point.
   *
   * @return The x coordinate.
  **/
  real& X(void);

  /**
   * Returns the x coordinate of this point.
   *
   * @return The x coordinate.
  **/
  real X(void) const;

  /**
   * Returns the y coordinate of this point.
   *
   * @return The y coordinate.
  **/
  real& Y(void);

  /**
   * Returns the y coordinate of this point.
   *
   * @return The y coordinate.
  **/
  real Y(void) const;

  /**
   * Returns the z coordinate of this point.
   *
   * @return The z coordinate.
  **/
  real& Z(void);

  /**
   * Returns the z coordinate of this point.
   *
   * @return The z coordinate.
  **/
  real Z(void) const;
  
	static axis::foundation::memory::RelativePointer Create(void);
	static axis::foundation::memory::RelativePointer Create(coordtype x, coordtype y);
	static axis::foundation::memory::RelativePointer Create(coordtype x, coordtype y, coordtype z);
	static axis::foundation::memory::RelativePointer Create(coordtype x, coordtype y, coordtype z, real weight);
private:
  axis::foundation::memory::RelativePointer state_;
  coordtype x_, y_, z_;
  real weight_;

  void *operator new(size_t bytes);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *ptr);
  void operator delete(void *, void *);
};

} } } // namespace axis::domain::integration
