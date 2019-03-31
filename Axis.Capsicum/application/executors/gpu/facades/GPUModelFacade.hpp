#pragma once
#include "domain/analyses/ModelOperatorFacade.hpp"
#include <list>
#include "services/scheduling/gpu/GPUTask.hpp"
#include "domain/fwd/finite_element_fwd.hpp"

namespace axis { namespace application { namespace executors { namespace gpu { 
  namespace facades {

/**
 * Implements a model operator facade that executes on GPU.
 * @sa ModelOperatorFacade
 */
class GPUModelFacade : public axis::domain::analyses::ModelOperatorFacade
{
public:  
  GPUModelFacade(void);
  ~GPUModelFacade(void);
  virtual void Destroy( void ) const;

  /**
   * Adds an element task which runs on elements of a specific type.
   *
   * @param [in,out] elementTask   The element task.
   * @param [in,out] elementSample One of the elements in the task that will
   *                 be used to extract its list of commands.
   */
  void AddElementTask(axis::services::scheduling::gpu::GPUTask& elementTask,
    axis::domain::elements::FiniteElement& elementSample);

  /**
   * Allocates, in the host and external processing device, memory necessary to
   * execute operations in this facade.
   */
  void AllocateMemory(void);

  /**
   * Deallocates memory claimed by this facade..
   */
  void DeallocateMemory(void);

  /**
   * Initializes model data in the host memory.
   */
  void InitMemory(void);

  /**
   * Mirrors model data and any supplementary information to the external
   * processing device memory.
   */
  void Mirror(void);

  /**
   * Copies back, from the external processing device, model data and any 
   * supplementary information so that any updates made externally are seen
   * by the host.
   */
  void Restore(void);

  /**
   * Looks up for the output bucket of model elements. This operation is
   * necessary for gather-vector/matrix operations to succeed.
   */
  void InitElementBuckets(void);

  /**
   * A synonym for the Restore method.
   * @sa Restore
   */
  virtual void RefreshLocalMemory( void );

  /**
   * Blocks calling thread until all asynchronous operations executed by this
   * facade terminates. Note that external device processing requests are 
   * all asynchronous.
   */
  virtual void Synchronize(void);

  virtual void CalculateGlobalLumpedMass( real t, real dt);
  virtual void CalculateGlobalConsistentMass( real t, real dt);
  virtual void CalculateGlobalStiffness( real t, real dt);
  virtual void CalculateGlobalInternalForce( real time, real lastTimeIncrement, 
    real nextTimeIncrement );
  virtual void UpdateStrain(real time, real lastTimeIncrement, 
    real nextTimeIncrement);
  virtual void UpdateStress(real time, real lastTimeIncrement, 
    real nextTimeIncrement);

  virtual void UpdateGeometry(real time, real lastTimeIncrement,
    real nextTimeIncrement);

  virtual void UpdateNodeQuantities( void );

private:
  class Tuple
  {
  public:
    Tuple(axis::services::scheduling::gpu::GPUTask& elementTask,
      axis::domain::formulations::FormulationStrategy&formulation,
      axis::domain::materials::MaterialStrategy &material,
      size_type formulationBlockSize, size_type materialBlockSize, 
      int dofCount);

    axis::services::scheduling::gpu::GPUTask *Task;
    axis::domain::formulations::FormulationStrategy *GPUFormulation;
    axis::domain::materials::MaterialStrategy *GPUMaterial;
    size_type FormulationBlockSize;
    size_type MaterialBlockSize;
    int TotalDofCount;
  };
  typedef std::list<Tuple> task_list;
  task_list tasks_;
};

} } } } } // namespace axis::application::executors::gpu::facades
