#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"
#include "foundation/blas/DenseMatrix.hpp"
#include "foundation/blas/SymmetricMatrix.hpp"
#include "foundation/blas/TriangularMatrix.hpp"
#include "foundation/blas/ColumnVector.hpp"
#include "foundation/blas/RowVector.hpp"

namespace axis { 

namespace services { 
namespace messaging {
class ResultMessage;
} // namespace axis::services::messaging
namespace diagnostics { namespace information {
class SolverCapabilities;
} } // namespace axis::services::diagnostics::information
}// namespace axis::services

namespace domain { 
namespace analyses {
class NumericalModel;
class AnalysisInfo;
} // namespace analyses

namespace elements {
class Node;
class FiniteElement;
} // namespace elements
} // namespace axis::domain::analyses


namespace application { 

namespace jobs {
class WorkFolder;
} // namespace axis::application::jobs

namespace output { 
  
namespace collectors {
class EntityCollector;

namespace messages {
class SnapshotStartMessage;
class SnapshotEndMessage;
} // namespace axis::application::output::collectors::messages

} // namespace axis::application::output::collectors


namespace recordsets {

/**
 * Represents a data set which stores results from an entity group
 * in the analysis numerical model.
 */
class AXISCOMMONLIBRARY_API ResultRecordset
{
public:

  /**
   * Values that specify recordset field types.
   */
  enum FieldType
  {
    IntegerNumber,
    RealNumber,
    Literal,
    Character,
    BooleanField,
    Undefined
  };

  virtual ~ResultRecordset(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * Prepares this recordset for writing operation.
   *
   * @param entitySet      Entity name to which this recordset is associated to. Empty string if none.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void OpenRecordset(const axis::String& entitySet);

  /**
   * Asks the recordset to perform any pending operation and then free resources.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void CloseRecordset(void);

  /**
   * Called before the recordset owner start creating fields in this recordset.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginCreateField(void);

  /**
   * Creates a data field in this recordset.
   *
   * @param fieldName Name of the field.
   * @param fieldType Data type of the field.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void CreateField(const axis::String& fieldName, FieldType fieldType);

  /**
   * Creates a data field capable of holding a sclar matrix.
   *
   * @param fieldName Name of the field.
   * @param rows      The number of rows in the matrix.
   * @param columns   The number of columns in the matrix.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void CreateMatrixField(const axis::String& fieldName, int rows, int columns);

  /**
   * Creates a data field capable of holding a scalar vector.
   *
   * @param fieldName Name of the field.
   * @param length    The vector length.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void CreateVectorField(const axis::String& fieldName, int length);

  /**
   * Called when the recordset owner finishes field creation.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void EndCreateField(void);

  /**
   * Prepares the recordset to write data for a new analysis step.
   *
   * @param stepName   Name of the step to which data is going to be written about.
   * @param stepIndex  Zero-based index of the step.
   * @param solverCaps Capabilities for the solver that will be employed.
   * 
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginAnalysisStep(const axis::String& stepName, int stepIndex, 
                    const axis::services::diagnostics::information::SolverCapabilities& solverCaps);

  /**
   * Asks the recordset to finish writing data about the last analysis step.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void EndAnalysisStep(void);

  /**
   * Prepares the recordset to write data for a new snapshot in the current analysis step.
   *
   * @param analysisInfo Current state of the analysis being run.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginSnapshot(const axis::domain::analyses::AnalysisInfo& analysisInfo);

  /**
   * Asks the recordset to finish writing data about the last snapshot.
   *
   * @param analysisInfo Current state of the analysis being run.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void EndSnapshot(const axis::domain::analyses::AnalysisInfo& analysisInfo);

  /**
   * Prepares the recordset to write data for a single node in the numerical model.
   *
   * @param message The message which triggered this request.
   * @param node    The subject node.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginNodeRecord(const axis::services::messaging::ResultMessage& message, 
                               const axis::domain::elements::Node& node);

  /**
   * Tells the recordset to finish writing data about the current node.
   *
   * @param message The message which triggered this request.
   * @param node    The subject node.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void EndNodeRecord(const axis::services::messaging::ResultMessage& message, 
                             const axis::domain::elements::Node& node);

  /**
   * Prepares the recordset to write data for a single element in the numerical model.
   *
   * @param message The message which triggered this request.
   * @param element The subject element.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginElementRecord(const axis::services::messaging::ResultMessage& message, 
                                  const axis::domain::elements::FiniteElement& element);

  /**
   * Tells the recordset to finish writing data about the current node.
   *
   * @param message The message which triggered this request.
   * @param element The subject element.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void EndElementRecord(const axis::services::messaging::ResultMessage& message, 
                                const axis::domain::elements::FiniteElement& element);

  /**
   * Prepares the recordset to write possible unordered and generic data from the numerical model state.
   *
   * @param message        The message which triggered this request.
   * @param numericalModel The analysis numerical model.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   */
  virtual void BeginGenericRecord(const axis::services::messaging::ResultMessage& message, 
                                  const axis::domain::analyses::NumericalModel& numericalModel);

  /**
   * Tells the recordset to finish writing data which came after the last call to BeginGenericRecord.
   *
   * @param message        The message which triggered this request.
   * @param numericalModel The analysis numerical model.
   *                       
   * @remark Overridable member. Base implementation does nothing.
   *                       
   * @sa BeginGenericRecord
   */
  virtual void EndGenericRecord(const axis::services::messaging::ResultMessage& message, 
                                const axis::domain::analyses::NumericalModel& numericalModel);

  /**
   * Initializes this recordset.
   *
   * @param [in,out] workFolder Location where output files should be written.
   */
  virtual void Init(axis::application::jobs::WorkFolder& workFolder);

  /**
   * Returns if this recordset is initialized and accepting to populate fields,
   * but not necessarily accepting write operations.
   *
   * @return true if it is, false otherwise.
   */
  virtual bool IsInitialized(void) const = 0;

  /**
   * Returns if this recordset is ready to accept write operations.
   *
   * @return true if it is, false otherwise.
   */
  virtual bool IsReady(void) const = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(int data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(real data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(bool data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::String& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::DenseMatrix& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::SymmetricMatrix& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::LowerTriangularMatrix& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::UpperTriangularMatrix& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::ColumnVector& data) = 0;

  /**
   * Writes data to a field in the current record.
   *
   * @param data The data.
   */
  virtual void WriteData(const axis::foundation::blas::RowVector& data) = 0;
}; // ResultRecordset

} } } } // namespace axis::application::output::recordsets
