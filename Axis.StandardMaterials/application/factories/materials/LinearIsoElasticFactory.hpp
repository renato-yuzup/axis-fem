#pragma once
#include "application/factories/materials/MaterialFactory.hpp"

namespace axis
{
	namespace application
	{
		namespace factories
		{
			namespace materials
			{
				class LinearIsoElasticFactory : public axis::application::factories::materials::MaterialFactory
				{
				public:
					LinearIsoElasticFactory(void);
					~LinearIsoElasticFactory(void);

					virtual bool CanBuild( const axis::String& modelName, const axis::services::language::syntax::evaluation::ParameterList& params ) const;

					virtual axis::domain::materials::MaterialModel& Build( const axis::String& modelName, const axis::services::language::syntax::evaluation::ParameterList& params );

					virtual void Destroy( void ) const;
				};
			
			}
		}
	}
}

