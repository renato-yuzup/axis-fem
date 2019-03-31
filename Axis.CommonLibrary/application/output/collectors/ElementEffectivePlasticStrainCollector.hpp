#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "ElementSetCollector.hpp"

namespace axis { namespace application { namespace output { namespace collectors {

/**
 * Collects element strain data, in a element-by-element basis.
 *
 * @sa ElementSetCollector
 */
class AXISCOMMONLIBRARY_API ElementEffectivePlasticStrainCollector : 
  public ElementSetCollector
{
public:
  /**
   * Creates a new element strain collector.
   *
   * @param targetSetName Element set name in which this collector will act.
   *
   * @return A new collector.
   */
  static ElementEffectivePlasticStrainCollector& Create(const axis::String& targetSetName);

  /**
   * Creates a new element strain collector.
   *
   * @param targetSetName   Element set name in which this collector will act.
   * @param customFieldName Field name to use in recordset.
   *
   * @return A new collector.
   */
  static ElementEffectivePlasticStrainCollector& Create(const axis::String& targetSetName,
                                                  const axis::String& customFieldName);

  virtual ~ElementEffectivePlasticStrainCollector(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy( void ) const;

  /**
   * Pushes into a recordset data collected from a node.
   *
   * @param element            The element from which collect data.
   * @param message            The message that triggered this request.
   * @param [in,out] recordset The recordset in which data will be fed.
   */
  virtual void Collect( const axis::domain::elements::FiniteElement& element,
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
  ElementEffectivePlasticStrainCollector(const axis::String& targetSetName, const axis::String& customFieldName);
  virtual axis::String GetFriendlyDescription( void ) const;
  axis::String fieldName_;
};

} } } } // namespace axis::application::output::collectors
