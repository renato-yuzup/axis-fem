/// <summary>
/// Contains definition for the class axis::domain::elements::FiniteElement.
/// </summary>
/// <author>Renato T. Yamassaki</author>

#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "foundation/Axis.SystemBase.hpp"
#include "foundation/collections/Collectible.hpp"
#include "domain/fwd/finite_element_fwd.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/blas/blas.hpp"

namespace axis { namespace domain { namespace elements {

/// <summary>
/// Represents a finite element in the model.
/// </summary>
/// <remarks>
/// The memory allocation model of every derived class from <see cref="FiniteElement" />
/// must be arena, so that no significant overhead is caused by memory allocation calls.
/// </remarks>
class AXISCOMMONLIBRARY_API FiniteElement
{
public:

  /**
   * Creates a new finite element with the specified geometry and behaves as 
   * described in the given formulation.
   *
   * @param id            Numerical identifier.
   * @param geometry      The geometry containing the nodes of this element.
   * @param materialModel The element material model.
   * @param formulation   An object which describes the behavior of this element.
  **/
	FiniteElement(id_type id, 
    const axis::foundation::memory::RelativePointer& geometry, 
    axis::domain::materials::MaterialModel& materialModel,
    axis::domain::formulations::Formulation& formulation);

  /**
   * Creates a new finite element with the specified geometry and behaves as 
   * described in the given formulation.
   *
   * @param internalId             Numerical identifier used in calculation 
   *                               process.
   * @param userId                 Numerical identifier by which this element is 
   *                               known to user and  external systems.
   * @param [in,out] geometry      The geometry containing the nodes of this 
   *                               element.
   * @param [in,out] materialModel The element material model.
   * @param [in,out] formulation   An object which describes the behavior of 
   *                               this element.
  **/
	FiniteElement(id_type internalId, id_type userId, 
    const axis::foundation::memory::RelativePointer& geometry, 
    axis::domain::materials::MaterialModel& materialModel,
    axis::domain::formulations::Formulation& formulation);

  /**
   * Destroys this object and frees all resources associated to it.
  **/
	~FiniteElement(void);

  /**
   * Destroys this object.
  **/
  void Destroy(void) const;

  /**
   * Returns the identifier by which this element is known within the analysis 
   * system.
   *
   * @return The identifier.
  **/
  id_type GetInternalId(void) const;

  /**
   * Returns the identifier by which user refers to this element.
   *
   * @return The user identifier.
  **/
  id_type GetUserId(void) const;

  /**
   * Returns the element geometry.
   *
   * @return The geometry.
  **/
	axis::domain::elements::ElementGeometry& Geometry(void);

  /**
   * Returns the element geometry.
   *
   * @return The geometry.
  **/
  const axis::domain::elements::ElementGeometry& Geometry(void) const;

  /**
   * Returns the constitutive model which describes the behavior of the element 
   * section.
   *
   * @return The material model.
  **/
	axis::domain::materials::MaterialModel& Material(void);

  /**
  * Returns the constitutive model which describes the behavior of the element 
  * section.
   *
   * @return The material model.
  **/
  const axis::domain::materials::MaterialModel& Material(void) const;

  /**
   * Returns the physical state of this element.
   *
   * @return The physical state.
  **/
  axis::domain::physics::InfinitesimalState& PhysicalState(void);

  /**
   * Returns the physical state of this element.
   *
   * @return The physical state.
  **/
  const axis::domain::physics::InfinitesimalState& PhysicalState(void) const;


  /**
   * Returns the last calculated stiffness matrix of this element.
   *
   * @return The positive-definite symmetric stiffness matrix.
  **/
	const axis::foundation::blas::SymmetricMatrix& GetStiffness(void) const;

  /**
   * Returns the last calculated consistent mass matrix of this element.
   *
   * @return The dense mass matrix.
  **/
	const axis::foundation::blas::SymmetricMatrix& GetConsistentMass(void) const;

  /**
   * Returns the last calculated lumped (diagonal) mass matrix of this element.
   *
   * @return A vector representing the main diagonal of the lumped mass matrix.
  **/
	const axis::foundation::blas::ColumnVector& GetLumpedMass(void) const;

  /**
   * Returns total accumulated artificial energy in this element, which is the
   * work done by anti-hourglass forces.
   *
   * @return The total artificial energy.
  **/
  real GetTotalArtificialEnergy(void) const;

  /**
   * Requests that the element allocates memory to store internal state data
   * for use throughout the analysis process.
  **/
	void AllocateMemory(void);

  /**
   * Requests that the element initializes its internal state and executes any
   * computation needed before starting analysis process.
  **/
	void CalculateInitialState(void);

  /**
   * Writes to a vector the local quantities related to this element obtained
   * from a model-wide vector field.
   *
   * @param [in,out] localField Vector where local quantities should be written.
   * @param modelField          The model vector field.
  **/
	void ExtractLocalField(axis::foundation::blas::ColumnVector& localField, 
    const axis::foundation::blas::ColumnVector& modelField) const;

  /**
   * Updates the element strain.
   *
   * @param elementDisplacementIncrement The displacement increment of this 
   *                                     element.
  **/
	void UpdateStrain(
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement);

  /**
   * Updates the element stress.
   *
   * @param elementDisplacementIncrement The displacement of this element.
   * @param elementVelocity              The velocity of this element.
   * @param timeInfo                   Current time state of the analysis.
  **/
	void UpdateStress(
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement, 
    const axis::foundation::blas::ColumnVector& elementVelocity,
    const axis::domain::analyses::AnalysisTimeline& timeInfo);

  /**
   * Calculates and writes in a vector the internal forces acting on this 
   * element.
   *
   * @param [in,out] internalForce        Vector where internal forces should 
   *                                      be written.
   * @param elementDisplacementIncrement  Element displacement increment.
   * @param elementVelocity               Element velocity field.
   * @param timeInfo                      Current time state of the analysis.
  **/
	void UpdateInternalForce(
    axis::foundation::blas::ColumnVector& internalForce, 
    const axis::foundation::blas::ColumnVector& elementDisplacementIncrement,
    const axis::foundation::blas::ColumnVector& elementVelocity,
    const axis::domain::analyses::AnalysisTimeline& timeInfo);

  /**
   * Requests the element to updates its internal state regarding its own
   * geometry.
   */
  void UpdateGeometry(void);

  /**
   * Requests to update one or some of the element matrices.
   *
   * @param whichMatrices     Object describing which matrices to calculate.
   * @param elementDisplacement The total displacement of the element.
   * @param elementVelocity     The element velocity.
  **/
	void UpdateMatrices(const MatrixOption& whichMatrices, 
    const axis::foundation::blas::ColumnVector& elementDisplacement,
    const axis::foundation::blas::ColumnVector& elementVelocity);

  /**
   * Requests to free all memory resources allocated by this element.
  **/
	void ClearMemory(void);

  /**
   * Returns the maximum timestep in a transient analysis that this element 
   * supports to ensure its numerical stability.
   *
   * @param elementDisplacement The element displacement.
   *
   * @return The critical timestep.
  **/
	real GetCriticalTimestep(
    const axis::foundation::blas::ColumnVector& elementDisplacement) const;

  /**
   * Returns if this element can be processed using CPU.
   *
   * @return true if CPU capable, false otherwise.
  **/
  bool IsCPUCapable(void) const;

  /**
   * Returns if this element can be processed using GPU.
   *
   * @return true if GPU capable, false otherwise.
  **/
  bool IsGPUCapable(void) const;

  /**
   * Returns the unique identifier for the formulation of this element.
   *
   * @return An universal unique identifier (UUID) which is exclusive among
   *         all formulation types.
  **/
  axis::foundation::uuids::Uuid GetFormulationTypeId(void) const;

  /**
   * Returns the unique identifier for the material model of this element.
   *
   * @return An universal unique identifier (UUID) which is exclusive among
   *         all material model types.
  **/
  axis::foundation::uuids::Uuid GetMaterialTypeId(void) const;

  /**
   * Returns the data length, in bytes, of formulation data required to run in 
   * GPU.
   *
   * @return The formulation data block size.
  **/
  size_type GetFormulationBlockSize(void) const;

  /**
   * Returns the data length, in bytes, of material data required to run in GPU.
   *
   * @return The material data block size.
  **/
  size_type GetMaterialBlockSize(void) const;

  /**
   * Initializes the GPU formulation data.
   *
   * @param [in,out] baseDataAddress Base memory address where formulation data 
   *                 should be written.
  **/
  void InitializeGPUFormulation(void *baseDataAddress);

  /**
   * Initializes the GPU material data.
   *
   * @param [in,out] baseDataAddress Base memory address where material data 
   *                 should be written.
  **/
  void InitializeGPUMaterial(void *baseDataAddress);

  axis::domain::formulations::FormulationStrategy& 
    GetGPUFormulationStrategy(void);
  axis::domain::materials::MaterialStrategy& GetGPUMaterialStrategy(void);

  static axis::foundation::memory::RelativePointer Create(
    id_type id, const axis::foundation::memory::RelativePointer& geometry, 
    axis::domain::materials::MaterialModel& materialModel,
    axis::domain::formulations::Formulation& formulation);
	static axis::foundation::memory::RelativePointer Create(
    id_type internalId, id_type userId, 
    const axis::foundation::memory::RelativePointer& geometry, 
    axis::domain::materials::MaterialModel& materialModel,
    axis::domain::formulations::Formulation& formulation);
  void *operator new(size_t bytes);
  void *operator new(size_t bytes, void *ptr);
  void operator delete(void *ptr);
  void operator delete(void *, void *);
private:
  id_type internalId_, externalId_;
  axis::domain::materials::MaterialModel *materialModel_; 
  axis::domain::formulations::Formulation *formulation_;	
  axis::foundation::memory::RelativePointer geometry_;
  axis::foundation::memory::RelativePointer physicalState_;
};

} } } // namespace axis::domain::elements
