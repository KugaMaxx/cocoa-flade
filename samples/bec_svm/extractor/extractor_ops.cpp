#include "extractor_ops.h"

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

PYBIND11_MODULE(extractor_ops, m) {
  using pybind11::operator""_a;

  py::class_<Converter>(m, "Converter")
      .def(py::init<const cv::Size &>(), "resolution"_a)
      .def("process", &Converter::process, "events"_a, "box"_a);
}
