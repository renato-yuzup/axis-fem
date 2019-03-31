#pragma once
#include "foundation/Axis.Core.hpp"
#include "AxisString.hpp"

namespace axis { namespace application { namespace jobs {

/**********************************************************************************************//**
  * <summary> Stores information required to execute a finite element
  * 			 analysis job.</summary>
  **************************************************************************************************/
class AXISCORE_API JobRequest
{
public:
  /**********************************************************************************************//**
    * <summary> Creates this object.</summary>
    *
    * <param name="masterFilename">   Filename of the master analysis file.</param>
    * <param name="baseIncludePath">  Full pathname of the base include
    * 								   folder.</param>
    * <param name="outputFolderPath"> Full pathname of the output
    * 								   folder for the result files.</param>
    *
    * <returns> A new instance of this class.</returns>
    **************************************************************************************************/
  JobRequest(const axis::String& masterFilename,
             const axis::String& baseIncludePath,
             const axis::String& outputFolderPath);

  /**********************************************************************************************//**
    * <summary> Destructor.</summary>
    **************************************************************************************************/
  ~JobRequest(void);

  /**********************************************************************************************//**
    * <summary> Destroys this object.</summary>
    **************************************************************************************************/
  void Destroy(void) const;

  /**********************************************************************************************//**
    * <summary> Returns the master analysis file path.</summary>
    *
    * <returns> The master analysis file path.</returns>
    **************************************************************************************************/
  axis::String GetMasterInputFilePath(void) const;

  /**********************************************************************************************//**
    * <summary> Returns the base include folder path.</summary>
    *
    * <returns> The base include path.</returns>
    **************************************************************************************************/
  axis::String GetBaseIncludePath(void) const;

  /**********************************************************************************************//**
    * <summary> Returns the output folder path.</summary>
    *
    * <returns> The output folder path.</returns>
    **************************************************************************************************/
  axis::String GetOutputFolderPath(void) const;

  /**********************************************************************************************//**
    * <summary> Adds a conditional flag for use when interpreting input files.</summary>
    *
    * <param name="flagName"> Flag name.</param>
    **************************************************************************************************/
  void AddConditionalFlag(const axis::String& flagName);

  /**********************************************************************************************//**
    * <summary> Clears all conditional flags.</summary>
    **************************************************************************************************/
  void ClearConditionalFlags(void);

  /**********************************************************************************************//**
    * <summary> Returns the conditional flags count.</summary>
    *
    * <returns> The conditional flags count.</returns>
    **************************************************************************************************/
  size_type GetConditionalFlagsCount(void) const;

  /**********************************************************************************************//**
    * <summary> Returns a conditional flag.</summary>
    *
    * <param name="index"> Zero-based index of the flag.</param>
    *
    * <returns> The conditional flag.</returns>
    **************************************************************************************************/
  axis::String GetConditionalFlag(size_type index) const;

  /**
    * Makes a deep copy of this object.
    *
    * @return A copy of this object.
    */
  JobRequest& Clone(void) const;
private:
  class Pimpl;
  Pimpl *pimpl_;

  // disallow copy constructor and copy assignment operator
  JobRequest(const JobRequest&);
  JobRequest& operator =(const JobRequest&);
};
    
} } } // namespace axis::application::jobs
