#pragma once

#define COMMON_SCHEDULE_TINY_OPS			schedule(static, 512)
#define COMMON_SCHEDULE_SMALL_OPS			schedule(static, 256)
#define COMMON_SCHEDULE_MEDIUM_OPS			schedule(static, 128)
#define COMMON_SCHEDULE_LARGE_OPS			schedule(static, 32)
#define COMMON_SCHEDULE_HUGE_OPS			schedule(dynamic, 4)
