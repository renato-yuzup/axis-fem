#pragma once

#define VECTOR_SCHEDULE_TINY_OPS			schedule(static,256)
#define VECTOR_SCHEDULE_MEDIUM_OPS		schedule(static,128)
#define MATRIX_SCHEDULE_TINY_OPS			schedule(static,256)
#define MATRIX_SCHEDULE_MEDIUM_OPS		schedule(static,128)

