project(NSC)

set(HEADER_FILES
        spkmeans.h
        )

set(SOURCE_FILES
        spkmeans.c
        )

add_library(src STATIC ${SOURCE_FILES} ${HEADER_FILES})