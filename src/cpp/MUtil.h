//
// Created by egrzrbr on 2019-04-06.
//



#ifndef CPP_EXAMPLE_MUTIL_H
#define CPP_EXAMPLE_MUTIL_H

#ifdef RESOURCES_DATA_PATH
#define PREPEND_PATH(f) RESOURCES_DATA_PATH + '/' + f

bool is_file_exist(const std::string &fileName) {

	return std::ifstream(PREPEND_PATH(fileName)).good();

}

#endif

#ifdef __clock_t_defined

clock_t TIME_0;

#define TIME_START TIME_0 = clock();
#define TIME_CHECK std::cout << "TIMEIT : " << clock() - TIME_0 << std::endl;

#define WAIT std::this_thread::sleep_for(std::chrono::milliseconds(1000));

#define obj_assert_ptr(obj) assert(std::is_pointer<decltype(obj)>::value !=0)
#define type_assert_ptr(type) assert(std::is_pointer<type>::value !=0)

#endif

std::string ts2h(const time_t rawtime) {

	struct tm *dt;
	char buffer[30];
	dt = localtime(&rawtime);
	strftime(buffer, sizeof(buffer), "%m%d%H%M%y", dt);
	return std::string(buffer);
}

enum debug_e {
	DEBUG0,
	DEBUG1,
	DEBUG2,
	DEBUG3,
	DEBUG4,
	DEBUG5
};


uint32_t flipBytes(uint32_t n) {

	uint32_t b0, b1, b2, b3;

	b0 = (n & 0x000000ff) << 24u;
	b1 = (n & 0x0000ff00) << 8u;
	b2 = (n & 0x00ff0000) >> 8u;
	b3 = (n & 0xff000000) >> 24u;

	return (b0 | b1 | b2 | b3);

}

#define D3(W, H, D, i, j, k) (i*H*D) + (j*D) + k

template<typename ...Args>
auto log(Args &&...args) {

	(std::cout << ... << args) << std::endl;
}

#endif //CPP_EXAMPLE_MUTIL_H
