#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "NodeSetCollector.hpp"
#include "Direction3DState.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

/**
 * Collects nodal acceleration data, in a node-by-node basis.
 *
 * @sa NodeSetCollector
 */
class AXISCOMMONLIBRARY_API NodeAccelerationCollector : public NodeSetCollector
{
public:

  /**
   * Creates a new node acceleration collector.
   *
   * @param targetSetName Node set name in which this collector will act.
   *
   * @return A new collector.
   */
  static NodeAccelerationCollector& Create(const axis::String& targetSetName);

  /**
   * Creates a new node acceleration collector.
   *
   * @param targetSetName   Node set name in which this collector will act.
   * @param customFieldName Field name to use in recordset.
   *
   * @return A new collector.
   */
  static NodeAccelerationCollector& Create(const axis::String& targetSetName,
                                           const axis::String& customFieldName);

  /**
   * Creates a new node acceleration collector.
   *
   * @param targetSetName Node set name in which this collector will act.
   * @param xState        Tells if X-component acceleration should be collected.
   * @param yState        Tells if Y-component acceleration should be collected.
   * @param zState        Tells if Z-component acceleration should be collected.
   *
   * @return A new collector.
   */
  static NodeAccelerationCollector& Create(const axis::String& targetSetName,
                                           XDirectionState xState, YDirectionState yState, 
                                           ZDirectionState zState);

  /**
   * Creates a new node acceleration collector.
   *
   * @param targetSetName   Node set name in which this collector will act.
   * @param customFieldName Field name to use in recordset.
   * @param xState          Tells if X-component acceleration should be collected.
   * @param yState          Tells if Y-component acceleration should be collected.
   * @param zState          Tells if Z-component acceleration should be collected.
   *
   * @return A new collector.
   */
  static NodeAccelerationCollector& Create(const axis::String& targetSetName,
                                           const axis::String& customFieldName,
                                           XDirectionState xState, YDirectionState yState, 
                                           ZDirectionState zState);
  
  virtual ~NodeAccelerationCollector(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy( void ) const;

  /**
   * Pushes into a recordset data collected from a node.
   *
   * @param node               The node from which collect data.
   * @param message            The message that triggered this request.
   * @param [in,out] recordset The recordset in which data will be fed.
   */
  virtual void Collect( const axis::domain::elements::Node& node,
                        const axis::services::messaging::ResultMessage& message, 
                        axis::application::output::recordsets::ResultRecordset& recordset );

  /**
   * Returns the name of the field that would be written on a recordset by this collector.
   *
   * @return The field name.
   */
  virtual axis::String GetFieldName( void ) const;

  /**********************************************************************************************//**
   * @brief Returns the data type that this collector captures.
   *
   * @return  The field type.
   **************************************************************************************************/
  virtual axis::application::output::DataType GetFieldType( void ) const;

  /**********************************************************************************************//**
   * @brief Returns how many positions the vector this collector writes has. If collector data 
   *        type is other than vector, this value must be equal to zero.
   *
   * @return  The vector field length.
   **************************************************************************************************/
  virtual int GetVectorFieldLength(void) const;
private:
  NodeAccelerationCollector(const axis::String& targetSetName, const axis::String& customFieldName);
  NodeAccelerationCollector(const axis::String& targetSetName, const axis::String& customFieldName,
                            XDirectionState xState, YDirectionState yState, ZDirectionState zState);

  virtual axis::String GetFriendlyDescription( void ) const;

  axis::String fieldName_;
  bool collectState_[3];
  int vectorLen_;
  real *values_;
};

} } } } // namespace axis::application::output::collectors
