#include "generator_ops.h"

namespace py = pybind11;

namespace pybind11::detail {

template<>
struct type_caster<cv::Size> {
	PYBIND11_TYPE_CASTER(cv::Size, _("tuple_xy"));

	bool load(handle obj, bool) {
		if (!py::isinstance<py::tuple>(obj)) {
			std::logic_error("Size(width,height) should be a tuple!");
			return false;
		}

		auto pt = reinterpret_borrow<py::tuple>(obj);
		if (pt.size() != 2) {
			std::logic_error("Size(width,height) tuple should be size of 2");
			return false;
		}

		value = cv::Size(pt[0].cast<int>(), pt[1].cast<int>());
		return true;
	}

	static handle cast(const cv::Size &resolution, return_value_policy, handle) {
		return py::make_tuple(resolution.width, resolution.height).release();
	}
};

} // namespace pybind11::detail

PYBIND11_MODULE(generator_ops, m) {
  using pybind11::operator""_a;

  py::class_<Detector>(m, "Detector")
      .def(py::init<const cv::Size &, float_t, size_t, float_t>(),
           "resolution"_a, "min_area"_a = 10.0, "candidate_num"_a = 5, "threshold"_a = 0.85)
      .def("detect", &Detector::detect, "events"_a);
}
