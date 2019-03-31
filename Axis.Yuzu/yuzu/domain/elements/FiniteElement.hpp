/// <summary>
/// Contains definition for the class axis::domain::elements::FiniteElement.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "yuzu/common/gpu.hpp"
#include "yuzu/domain/physics/InfinitesimalState.hpp"
#include "yuzu/foundation/memory/RelativePointer.hpp"
#include "yuzu/foundation/blas/ColumnVector.hpp"
#include "yuzu/foundation/blas/SymmetricMatrix.hpp"

namespace axis { namespace yuzu { namespace domain { namespace elements {

class ElementGeometry;

/// <summary>
/// Represents a finite element in the model.
/// </summary>
/// <remarks>
/// The memory allocation model of every derived class from <see cref="FiniteElement" />
/// must be arena, so that no significant overhead is caused by memory allocation calls.
/// </remarks>
class FiniteElement
{
public:
  /**
   * Destroys this object and frees all resources associated to it.
  **/
	GPU_ONLY ~FiniteElement(void);

  /**
   * Returns the internal identifier.
   *
   * @return The identifier by which this element is known during analysis process.
  **/
  GPU_ONLY id_type GetInternalId(void) const;

  /**
   * Returns the user identifier.
   *
   * @return The user identifier.
  **/
  GPU_ONLY id_type GetUserId(void) const;

  /**
   * Returns the element geometry.
   *
   * @return The geometry.
  **/
	GPU_ONLY axis::yuzu::domain::elements::ElementGeometry& Geometry(void);

  /**
   * Returns the element geometry.
   *
   * @return The geometry.
  **/
  GPU_ONLY const axis::yuzu::domain::elements::ElementGeometry& Geometry(void) const;

  /**
   * Returns the set of variables that describes the physical state of this element.
   *
   * @return The set of physical variables.
  **/
  GPU_ONLY axis::yuzu::domain::physics::InfinitesimalState& PhysicalState(void);

  /**
   * Returns the set of variables that describes the physical state of this element.
   *
   * @return The set of physical variables.
  **/
  GPU_ONLY const axis::yuzu::domain::physics::InfinitesimalState& PhysicalState(void) const;

  /**
   * Writes to a vector the quantities related to this element of a model-wide field.
   *
   * @param [in,out] localField Vector where local quantities should be written.
   * @param modelField          The model field.
  **/
	GPU_ONLY void ExtractLocalField(axis::yuzu::foundation::blas::ColumnVector& localField, 
                                  const axis::yuzu::foundation::blas::ColumnVector& modelField) const;
private:
  id_type internalId_, externalId_;
  void *materialModel_; // material data and operations
  void *formulation_;	// formulation (how it's done)
  axis::yuzu::foundation::memory::RelativePointer geometry_;			// geometry data (nodes, faces, ...)
  axis::yuzu::foundation::memory::RelativePointer physicalState_;

  GPU_ONLY FiniteElement(void);
  FiniteElement(const FiniteElement&);
  FiniteElement& operator =(const FiniteElement&);
};

} } } } // namespace axis::yuzu::domain::elements
