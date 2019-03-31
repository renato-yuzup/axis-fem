#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "services/messaging/ResultMessage.hpp"

namespace axis { namespace domain { 
  
namespace analyses {
class ModelKinematics;
class ModelDynamics;
} // namespace axis::domain::analyses

namespace algorithms { namespace messages {

/**
 * Message dispatched when a new state for the numerical model has been calculated.
 *
 * @sa axis::services::messaging::ResultMessage
 */
class AXISCOMMONLIBRARY_API ModelStateUpdateMessage : public axis::services::messaging::ResultMessage
{
public:

  /**
   * Constructor.
   *
   * @param kinematicState Kinematic state of the model mesh.
   * @param dynamicState   Dynamic state of the model mesh.
   */
  ModelStateUpdateMessage(const axis::domain::analyses::ModelKinematics& kinematicState,
                          const axis::domain::analyses::ModelDynamics& dynamicState);

  /**
   * Destructor.
   */
  virtual ~ModelStateUpdateMessage(void);

  /**
   * Executes the destroy operation.
   */
  virtual void DoDestroy( void ) const;

  /**
   * Executes the clone operation.
   *
   * @param id The message identifier.
   *
   * @return A shallow copy of this message.
   */
  virtual axis::services::messaging::Message& DoClone( id_type id ) const;

  /**
   * Returns the mesh kinematic state.
   *
   * @return The mesh kinematic state.
   */
  const axis::domain::analyses::ModelKinematics& GetMeshKinematicState(void) const;

  /**
   * Returns the mesh dynamic state.
   *
   * @return The mesh dynamic state.
   */
  const axis::domain::analyses::ModelDynamics& GetMeshDynamicState(void) const;

  /**
   * Checks if 'message' is of this message type.
   *
   * @param message The message to verify.
   *
   * @return true if it is of kind, false otherwise.
   */
  static bool IsOfKind(const axis::services::messaging::Message& message);

  /**
   * Numerical identifier for this message type.
   */
  static const int MessageId;
private:
  const axis::domain::analyses::ModelKinematics& kinematicState_;
  const axis::domain::analyses::ModelDynamics& dynamicState_;
};

} } } } // namespace axis::domain::algorithms::messages
