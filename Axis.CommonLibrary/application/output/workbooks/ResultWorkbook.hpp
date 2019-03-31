#pragma once
#include "foundation/Axis.CommonLibrary.hpp"
#include "AxisString.hpp"
#include "foundation/uuids/Uuid.hpp"
#include "foundation/date_time/Timestamp.hpp"

namespace axis { 

namespace services { namespace diagnostics { namespace information {
class SolverCapabilities;
} } } // namespace axis::services::diagnostics::information

namespace domain { namespace analyses {
class AnalysisInfo;
class NumericalModel;
class AnalysisTimeline;
}} // namespace axis::domain::analyses

namespace application { 

namespace jobs {
class AnalysisStepInformation;
class WorkFolder;
} // namespace axis::application::jobs
  
namespace output { 

namespace recordsets {
class ResultRecordset;
} // namespace axis::application::output::recordsets
  
namespace workbooks {

/**
 * Manages result data for every entity set and other objects 
 * in the numerical model.
 */
class AXISCOMMONLIBRARY_API ResultWorkbook
{
public:
  ResultWorkbook(void);
  virtual ~ResultWorkbook(void);

  /**
   * Destroys this object.
   */
  virtual void Destroy(void) const = 0;

  /**
   * Creates and initializes the registered recordsets in this workbook.
   *
   * @param [in,out] workFolder Pathname of the analysis work folder.
   */
  void InitWorkbook(axis::application::jobs::WorkFolder& workFolder);

  /**
   * Prepares the workbook to write information about a new step.
   *
   * @param stepInfo Object containing information about the current analysis step.
   */
  void Open(const axis::application::jobs::AnalysisStepInformation& stepInfo);

  /**
   * Queries if this workbook is active and accepting data operations.
   *
   * @return true if open, false if not.
   */
  bool IsOpen(void) const;

  /**
   * Closes this workbook freeing any resource allocated for data operation.
   * As a consequence, data operations will no longer be accepted.
   */
  void Close(void);

  /**
   * Requests all recordsets to start a new analysis step.
   *
   * @param stepInfo Object containing information about the current analysis step.
   */
  void BeginStep(const axis::application::jobs::AnalysisStepInformation& stepInfo);

  /**
   * Requests all recordsets to end/close current analysis step.
   */
  void EndStep(void);

  /**
   * @brief Creates a new recordset for a node set.
   *
   * @param nodeSetName Name of the node set.
  **/
  void CreateNodeRecordset(const axis::String& nodeSetName);

  /**
   * @brief Creates a new recordset for an element set.
   *
   * @param elementSetName Name of the element set.
  **/
  void CreateElementRecordset(const axis::String& elementSetName);

  /**
   * Returns the recordset which contains result data from nodes of a specific set.
   *
   * @param nodeSetName Name of the node set associated to the recordset.
   *
   * @return The recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetNodeRecordset(const axis::String& nodeSetName);

  /**
   * Returns the recordset which contains result data from nodes of a specific set.
   *
   * @param index Zero-based index of the recordset.
   *
   * @return The recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetNodeRecordset(int index);

  /**
   * Returns node recordset count.
   *
   * @return The node recordset count.
   */
  int GetNodeRecordsetCount(void) const;

  /**
   * Returns the recordset which contains result data from elements of a specific set.
   *
   * @param elementSetName Name of the element set.
   *
   * @return The element recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetElementRecordset(const axis::String& elementSetName);

  /**
   * Returns the recordset which contains result data from elements of a specific set.
   *
   * @param index Zero-based index of the recordset.
   *
   * @return The element recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetElementRecordset(int index);

  /**
   * Returns element recordset count.
   *
   * @return The element recordset count.
   */
  int GetElementRecordsetCount(void) const;

  /**
   * Returns the recordset which stores generic result data.
   *
   * @return The generic recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetGenericSetRecordset(void);

  /**
   * Returns the main recordset which contains main simulation data or is responsible to merge
   * all other recordset data, depending on workbook implementation.
   *
   * @return The main recordset.
   */
  axis::application::output::recordsets::ResultRecordset& GetMainRecordset(void);

  /**
   * @brief Prepares the workbook to write information about an instant in the analysis.
   *
   * @param info  Information about the analysis.
  **/
  void BeginSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * @brief Tells the workbook to finish writing information about the current analysis instant.
   *
   * @param info  Information about the analysis.
  **/
  void EndSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * Returns the output name (generally used as the filename) assigned to this workbook.
   *
   * @return The workbook output name.
   */
  axis::String GetWorkbookOutputName(void) const;

  /**
   * Sets the output name (generally used as the filename) for this workbook.
   *
   * @param name The output name.
   */
  void SetWorkbookOutputName(const axis::String& name);

  /**
   * Toggles if the workbook should append data to the end of the file or clear it before using.
   *
   * @param appendState true to set append state on, false otherwise.
   */
  void ToggleAppendOperation(bool appendState);

  /**
   * Returns if this workbook is writing data to output in append operation.
   *
   * @return true if append operation is on, false otherwise.
   */
  bool IsAppendOperation(void) const;

  /**
   * Queries if we this workbook is able to work with output files in append operation.
   *
   * @return true if it can, false otherwise.
   */
  virtual bool SupportsAppendOperation(void) const = 0;

  /**
   * Returns a literal identifier for this workbook type, which should be exactly the file extension
   * this workbook uses.
   *
   * @return The format identifier.
   */
  virtual axis::String GetFormatIdentifier(void) const = 0;

  /**
   * Returns the friendly name of the format for this workbook.
   *
   * @return The format title.
   */
  virtual axis::String GetFormatTitle(void) const = 0;

  /**
   * Returns a short and concise description of the format for this workbook.
   *
   * @return The short description.
   */
  virtual axis::String GetShortDescription(void) const = 0;

  /**
   * @brief Returns if this workbook supports node recordsets.
   *
   * @return  true if it does, false otherwise.
  **/
  virtual bool SupportsNodeRecordset(void) const = 0;

  /**
   * @brief Returns if this workbook supports element recordsets.
   *
   * @return  true if it does, false otherwise.
  **/
  virtual bool SupportsElementRecordset(void) const = 0;

  /**
   * Returns if we this workbook supports the use of generic recordset.
   *
   * @return true if it supports, false otherwise.
   */
  virtual bool SupportsGenericRecordset(void) const = 0;

  /**
   * Returns if we this workbook supports the use of a main recordset.
   *
   * @return true if it supports, false otherwise.
   */
  virtual bool SupportsMainRecordset(void) const = 0;
private:
  /**
   * @brief Implementation-defined method to let the workbook execute actions before
   *        recordsets are opened.
   *
   * @param currentStep The current step.
  **/
  virtual void DoBeforeInit(axis::application::jobs::WorkFolder& workFolder);

  /**
   * @brief Implementation-defined method to let the workbook execute actions after
   *        recordsets are opened.
   *
   * @param currentStep The current step.
  **/
  virtual void DoAfterInit(axis::application::jobs::WorkFolder& workFolder);

  /**
   * Implementation-defined method to let the workbook execute actions before recordsets are opened.
   *
   * @param stepInfo Object containing information about the current analysis step.
   */
  virtual void DoBeforeOpen(const axis::application::jobs::AnalysisStepInformation& stepInfo);

  /**
   * Implementation-defined method to let the workbook execute actions after recordsets are opened.
   *
   * @param stepInfo Object containing information about the current analysis step.
   */
  virtual void DoAfterOpen(const axis::application::jobs::AnalysisStepInformation& stepInfo);

  /**
   * @brief Implementation-defined method to let the workbook execute actions before
   *        recordsets are closed.
   *
   * @param currentStep The current step.
  **/
  virtual void DoBeforeClose(void);

  /**
   * @brief Implementation-defined method to let the workbook execute actions after
   *        recordsets are closed.
   *
   * @param currentStep The current step.
  **/
  virtual void DoAfterClose(void);

  /**
   * @brief Implementation-defined method to let the workbook execute actions before
   *        recordsets are notified about the start of a new analysis instant.
   *
   * @param currentStep Information about the analysis.
  **/
  virtual void DoBeforeBeginSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * @brief Implementation-defined method to let the workbook execute actions after
   *        recordsets are notified about the start of a new analysis instant.
   *
   * @param currentStep Information about the analysis.
  **/
  virtual void DoAfterBeginSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * @brief Implementation-defined method to let the workbook execute actions before
   *        recordsets are notified about the end of the analysis instant.
   *
   * @param currentStep Information about the analysis.
  **/
  virtual void DoBeforeEndSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * @brief Implementation-defined method to let the workbook execute actions after
   *        recordsets are notified about the end of the analysis instant.
   *
   * @param currentStep Information about the analysis.
  **/
  virtual void DoAfterEndSnapshot(const axis::domain::analyses::AnalysisInfo& info);

  /**
   * @brief Requests the workbook to create an appropriate node recordset.
   *
   * @param nodeSetName Name of the node set associated to the recordset.
   *
   * @return  A new recordset.
  **/
  virtual axis::application::output::recordsets::ResultRecordset& 
                          DoCreateNodeRecordset(const axis::String& nodeSetName);

  /**
   * @brief Requests the workbook to create an appropriate element recordset.
   *
   * @param elementSetName Name of the element set associated to the recordset.
   *
   * @return  A new recordset.
  **/
  virtual axis::application::output::recordsets::ResultRecordset& 
                          DoCreateElementRecordset(const axis::String& elementSetName);

  /**
   * Requests to create a generic recordset for this workbook.
   *
   * @param [in,out] workFolder The analysis work folder.
   *
   * @return A new generic recordset.
   */
  virtual axis::application::output::recordsets::ResultRecordset& 
                          DoCreateGenericRecordset(axis::application::jobs::WorkFolder& workFolder);

  /**
   * Requests to create a generic recordset for this workbook.
   *
   * @param [in,out] workFolder The analysis work folder.
   *
   * @return A new generic recordset.
   */
  virtual axis::application::output::recordsets::ResultRecordset& 
                          DoCreateMainRecordset(axis::application::jobs::WorkFolder& workFolder);

  /**
   * Queries if this workbook is ready.
   *
   * @return true if it is, false otherwise.
   */
  virtual bool IsReady(void) const;

  class Pimpl;
  Pimpl *pimpl_;
}; // ResultWorkbook

} } } } // namespace axis::application::output::workbooks
