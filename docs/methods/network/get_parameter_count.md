# get_parameter_count()

Calculate total trainable parameters in the network.

**Class**: NeuralNetworkProxy

**Access**: `network.get_parameter_count()`

## Parameters

None

## Returns

Integer parameter count

## Example

```javascript
var params = network.get_parameter_count();

context.log("Network has " + params + " trainable parameters");
// Network has 54280 trainable parameters
```

## Calculation

Sums all weight matrices and bias vectors across all nodes

## See Also

- [generate_mlx_code()](generate_mlx_code.md) - Generate code
- [get_node()](get_node.md) - Get node info
