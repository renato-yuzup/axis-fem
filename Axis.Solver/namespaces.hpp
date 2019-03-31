/// <summary>
/// Contains namespaces definitions and its respectively description. 
/// </summary>

#pragma once

/// <summary>
/// The main namespace container of the application.
/// </summary>
namespace axis
{

	/// <summary>
	/// Contains classes that interact with the business logic of the application and
	/// work cooperatively with classes from the problem domain.
	/// </summary>
	namespace application
	{
		/// <summary>
		/// Contains classes specialized in the creation of classes.
		/// </summary>	
		namespace factories
		{
			/// <summary>
			/// Contains the implementation of the base class factories.
			/// </summary>	
			namespace Impl
			{

			}
		}

	}

	/// <summary>
	/// Contains classes derived from the domain model and also its implementations.
	/// </summary>
	namespace domain
	{
		/// <summary>
		/// Contains classes that define boundary conditions of a problem.
		/// </summary>	
		namespace boundary_conditions
		{
		}

		/// <summary>
		/// Contains classes that define the behavior of a material.
		/// </summary>	
		namespace Material
		{
		}

		/// <summary>
		/// Contains classes that compose the basic structure of a finite element.
		/// </summary>	
		namespace elements
		{
		}

		/// <summary>
		/// Contains classes that define the behavior of a finite element.
		/// </summary>	
		namespace formulations
		{
		}

		/// <summary>
		/// Contains classes that encapsulates logic for numerical integration process.
		/// </summary>	
		namespace integration
		{
		}

		/// <summary>
		/// Contains definitions of shape functions used by finite elements.
		/// </summary>	
		namespace shape_functions
		{
		}

	}

	/// <summary>
	/// Contains classes which delivers basic services needed to run basic tasks of the program.
	/// </summary>
	namespace services
	{
		/// <summary>
		/// Contains auxiliary classes and definitions for reading, writing and parsing configuration
		/// data for the application.
		/// </summary>
		namespace configuration
		{

		}

		/// <summary>
		/// Contains auxiliary classes and definitions for services to locate files and folders needed
		/// by the axis application.
		/// </summary>
		namespace io
		{

		}
	}

	/// <summary>
	/// Provides common services used by most part of the application.
	/// </summary>
	namespace foundation
	{
	}
}
