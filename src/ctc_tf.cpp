#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "ctc.h"

using namespace tensorflow;

REGISTER_OP("CTC")
    .Input("y: float")
    .Input("l: int32")
    .Output("loss: float")
    .Output("grad: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle y_input;
        ::tensorflow::shape_inference::ShapeHandle l_input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &y_input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &l_input));

        ::tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(y_input, 0);
        c->set_output(0, c->Vector(batch_size));
        c->set_output(1, y_input);
        return Status::OK();
    });

REGISTER_OP("CTCLossOnly")
    .Input("y: float")
    .Input("l: int32")
    .Output("loss: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle y_input;
        ::tensorflow::shape_inference::ShapeHandle l_input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &y_input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &l_input));

        ::tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(y_input, 0);
        c->set_output(0, c->Vector(batch_size));
        return Status::OK();
    });

REGISTER_OP("CTCEquals")
    .Input("y: float")
    .Input("l: int32")
    .Output("equals: uint8")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle y_input;
        ::tensorflow::shape_inference::ShapeHandle l_input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &y_input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &l_input));

        ::tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(y_input, 0);
        c->set_output(0, c->Vector(batch_size));
        return Status::OK();
    });

REGISTER_OP("CTCCer")
    .Input("y: float")
    .Input("l: int32")
    .Output("error_rates: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle y_input;
        ::tensorflow::shape_inference::ShapeHandle l_input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &y_input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &l_input));

        ::tensorflow::shape_inference::DimensionHandle batch_size = c->Dim(y_input, 0);
        c->set_output(0, c->Vector(batch_size));
        return Status::OK();
    });


class CTCOp : public OpKernel {

public:
    explicit CTCOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& y_tensor = context->input(0);
        auto y = y_tensor.flat<float>();
        const Tensor& l_tensor = context->input(1);
        auto l = l_tensor.flat<int32>();

        auto y_tensor_shape = y_tensor.shape();
        const auto n_batches = y_tensor_shape.dim_size(0);
        const auto timesteps = y_tensor_shape.dim_size(1);
        const auto alphabet_size = y_tensor_shape.dim_size(2);
        const auto labels_length = l_tensor.shape().dim_size(1);

        // Create output tensors
        Tensor* loss_tensor = nullptr;
        Tensor* grad_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {n_batches}, &loss_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, y_tensor.shape(), &grad_tensor));

        ctc(
            y.data(),
            reinterpret_cast<const unsigned*>(l.data()),
            n_batches,
            timesteps,
            alphabet_size,
            labels_length,
            loss_tensor->flat<float>().data(),
            grad_tensor->flat<float>().data()
        );
    }
};


class CTCLossOnlyOp : public OpKernel {

public:
    explicit CTCLossOnlyOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& y_tensor = context->input(0);
        auto y = y_tensor.flat<float>();
        const Tensor& l_tensor = context->input(1);
        auto l = l_tensor.flat<int32>(); 
        auto y_tensor_shape = y_tensor.shape();
        const auto n_batches = y_tensor_shape.dim_size(0);
        const auto timesteps = y_tensor_shape.dim_size(1);
        const auto alphabet_size = y_tensor_shape.dim_size(2);
        const auto labels_length = l_tensor.shape().dim_size(1);

        // Create output tensors
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {n_batches}, &output_tensor));

        ctc_loss_only(
            y.data(),
            reinterpret_cast<const unsigned*>(l.data()),
            n_batches,
            timesteps,
            alphabet_size,
            labels_length,
            output_tensor->flat<float>().data()
        );
    }
};


class CTCEqualsOp : public OpKernel {

public:
    explicit CTCEqualsOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& y_tensor = context->input(0);
        auto y = y_tensor.flat<float>();
        const Tensor& l_tensor = context->input(1);
        auto l = l_tensor.flat<int32>(); 
        auto y_tensor_shape = y_tensor.shape();
        const auto n_batches = y_tensor_shape.dim_size(0);
        const auto timesteps = y_tensor_shape.dim_size(1);
        const auto alphabet_size = y_tensor_shape.dim_size(2);
        const auto labels_length = l_tensor.shape().dim_size(1);

        // Create output tensors
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {n_batches}, &output_tensor));

        equals(
            y.data(),
            reinterpret_cast<const unsigned*>(l.data()),
            n_batches,
            timesteps,
            alphabet_size,
            labels_length,
            output_tensor->flat<unsigned char>().data()
        );
    }
};


class CTCCharacterErrorRateOp : public OpKernel {

public:
    explicit CTCCharacterErrorRateOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& y_tensor = context->input(0);
        auto y = y_tensor.flat<float>();
        const Tensor& l_tensor = context->input(1);
        auto l = l_tensor.flat<int32>(); 
        auto y_tensor_shape = y_tensor.shape();
        const auto n_batches = y_tensor_shape.dim_size(0);
        const auto timesteps = y_tensor_shape.dim_size(1);
        const auto alphabet_size = y_tensor_shape.dim_size(2);
        const auto labels_length = l_tensor.shape().dim_size(1);

        // Create output tensors
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {n_batches}, &output_tensor));

        character_error_rate(
            y.data(),
            reinterpret_cast<const unsigned*>(l.data()),
            n_batches,
            timesteps,
            alphabet_size,
            labels_length,
            output_tensor->flat<float>().data()
        );
    }
};


REGISTER_KERNEL_BUILDER(Name("CTC").Device(DEVICE_CPU), CTCOp);
REGISTER_KERNEL_BUILDER(Name("CTCLossOnly").Device(DEVICE_CPU), CTCLossOnlyOp);
REGISTER_KERNEL_BUILDER(Name("CTCEquals").Device(DEVICE_CPU), CTCEqualsOp);
REGISTER_KERNEL_BUILDER(Name("CTCCer").Device(DEVICE_CPU), CTCCharacterErrorRateOp);
