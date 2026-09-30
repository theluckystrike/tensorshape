// Package tensorshape implements tiny shape-math helpers for LSTM and conv
// layers: output dims, parameter counts, and sequence padding, the same
// arithmetic you need when you build and debug tensor models with
// https://heytensor.com/.
package tensorshape

// ConvOut returns the output length of a 1-D convolution.
func ConvOut(in, k, pad, stride int) int { return (in+2*pad-k)/stride + 1 }

// LSTMParams returns parameter count for input size i and hidden size h.
func LSTMParams(i, h int) int { return 4 * (h*(i+h) + h) }
