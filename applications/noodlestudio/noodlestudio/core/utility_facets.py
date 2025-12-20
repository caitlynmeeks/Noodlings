"""
Utility Facets - Common operations for agentic workflows

These are lightweight, deterministic facets that perform simple operations
without LLM calls. Useful for:
- Data transformation and routing
- Control flow (gates, branches, counters)
- Math operations (arithmetic, min/max, clamp)
- String manipulation (concat, split, format)
- Array operations (get, join, filter)

These can be chained together to build complex data processing pipelines.

Author: Caitlyn + Claude
Date: December 20, 2025
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import re
import json


# =============================================================================
# MATH FACETS
# =============================================================================

class MathAddFacet:
    """
    Add two numbers together.

    Inputs: a, b
    Outputs: result (a + b)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 0))
        return {'result': a + b, 'out': a + b}


class MathSubtractFacet:
    """
    Subtract b from a.

    Inputs: a, b
    Outputs: result (a - b)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 0))
        return {'result': a - b, 'out': a - b}


class MathMultiplyFacet:
    """
    Multiply two numbers.

    Inputs: a, b
    Outputs: result (a * b)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 1))
        return {'result': a * b, 'out': a * b}


class MathDivideFacet:
    """
    Divide a by b.

    Inputs: a, b
    Outputs: result (a / b), error (if division by zero)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 1))

        if b == 0:
            return {'result': 0, 'out': 0, 'error': 'Division by zero'}

        return {'result': a / b, 'out': a / b}


class MathMinFacet:
    """
    Return minimum of two values.

    Inputs: a, b
    Outputs: result (min(a, b))
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 0))
        result = min(a, b)
        return {'result': result, 'out': result}


class MathMaxFacet:
    """
    Return maximum of two values.

    Inputs: a, b
    Outputs: result (max(a, b))
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = float(inputs.get('a', 0))
        b = float(inputs.get('b', 0))
        result = max(a, b)
        return {'result': result, 'out': result}


class MathClampFacet:
    """
    Clamp value to range [min, max].

    Inputs: value, min, max
    Outputs: result (clamped value)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = float(inputs.get('value', 0))
        min_val = float(inputs.get('min', 0))
        max_val = float(inputs.get('max', 1))
        result = max(min_val, min(max_val, value))
        return {'result': result, 'out': result}


class MathAbsFacet:
    """
    Return absolute value.

    Inputs: value
    Outputs: result (|value|)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = float(inputs.get('value', inputs.get('in', 0)))
        result = abs(value)
        return {'result': result, 'out': result}


# =============================================================================
# LOGIC FACETS
# =============================================================================

class LogicAndFacet:
    """
    Logical AND of two boolean values.

    Inputs: a, b (truthy/falsy values)
    Outputs: result (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = bool(inputs.get('a', False))
        b = bool(inputs.get('b', False))
        result = a and b
        return {'result': result, 'out': result}


class LogicOrFacet:
    """
    Logical OR of two boolean values.

    Inputs: a, b (truthy/falsy values)
    Outputs: result (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = bool(inputs.get('a', False))
        b = bool(inputs.get('b', False))
        result = a or b
        return {'result': result, 'out': result}


class LogicNotFacet:
    """
    Logical NOT of a boolean value.

    Inputs: value (truthy/falsy)
    Outputs: result (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = bool(inputs.get('value', inputs.get('in', False)))
        result = not value
        return {'result': result, 'out': result}


class LogicCompareFacet:
    """
    Compare two values with configurable operator.

    Config: operator in prompt field ('==', '!=', '>', '<', '>=', '<=')
    Inputs: a, b
    Outputs: result (boolean)
    """

    def __init__(self, facet_id: str, operator: str = '=='):
        self.facet_id = facet_id
        self.operator = operator

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        a = inputs.get('a', 0)
        b = inputs.get('b', 0)

        # Try to compare as numbers if possible
        try:
            a = float(a)
            b = float(b)
        except (ValueError, TypeError):
            pass  # Keep as original types

        operators = {
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y,
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
        }

        op_func = operators.get(self.operator, operators['=='])
        result = op_func(a, b)
        return {'result': result, 'out': result}


class LogicSwitchFacet:
    """
    Route value based on condition (if/else).

    Inputs: condition (boolean), true_value, false_value
    Outputs: result (true_value if condition else false_value)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        condition = bool(inputs.get('condition', False))
        true_value = inputs.get('true_value', inputs.get('true', ''))
        false_value = inputs.get('false_value', inputs.get('false', ''))

        result = true_value if condition else false_value
        return {'result': result, 'out': result}


# =============================================================================
# STRING FACETS
# =============================================================================

class StringConcatFacet:
    """
    Concatenate strings.

    Inputs: a, b (or multiple inputs named in0, in1, in2...)
    Config: separator in prompt field (default: '')
    Outputs: result (concatenated string)
    """

    def __init__(self, facet_id: str, separator: str = ''):
        self.facet_id = facet_id
        self.separator = separator

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        # Collect all inputs
        parts = []

        # Check for a, b pattern
        if 'a' in inputs:
            parts.append(str(inputs['a']))
        if 'b' in inputs:
            parts.append(str(inputs['b']))

        # Check for numbered inputs (in0, in1, in2...)
        i = 0
        while f'in{i}' in inputs:
            parts.append(str(inputs[f'in{i}']))
            i += 1

        # Fall back to 'in' if nothing else
        if not parts and 'in' in inputs:
            parts.append(str(inputs['in']))

        result = self.separator.join(parts)
        return {'result': result, 'out': result}


class StringSplitFacet:
    """
    Split string by delimiter.

    Inputs: value (string to split)
    Config: delimiter in prompt field (default: ' ')
    Outputs: result (list of strings), first, last, count
    """

    def __init__(self, facet_id: str, delimiter: str = ' '):
        self.facet_id = facet_id
        self.delimiter = delimiter

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '')))
        parts = value.split(self.delimiter)

        return {
            'result': parts,
            'out': parts,
            'first': parts[0] if parts else '',
            'last': parts[-1] if parts else '',
            'count': len(parts)
        }


class StringReplaceFacet:
    """
    Replace substring in string.

    Inputs: value (string), search (substring to find), replace (replacement)
    Outputs: result (modified string)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '')))
        search = str(inputs.get('search', ''))
        replace = str(inputs.get('replace', ''))

        result = value.replace(search, replace)
        return {'result': result, 'out': result}


class StringFormatFacet:
    """
    Format string with placeholders.

    Inputs: template (string with {key} placeholders), plus any placeholder values
    Outputs: result (formatted string)

    Example: template="{name} is {age}" with name="Alice", age=30
             -> "Alice is 30"
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        template = str(inputs.get('template', inputs.get('in', '')))

        # All other inputs become format values
        format_values = {k: v for k, v in inputs.items() if k not in ('template', 'in')}

        try:
            result = template.format(**format_values)
        except KeyError as e:
            result = f"[Missing key: {e}]"

        return {'result': result, 'out': result}


class StringLengthFacet:
    """
    Get length of string.

    Inputs: value (string)
    Outputs: result (integer length)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '')))
        result = len(value)
        return {'result': result, 'out': result}


class StringContainsFacet:
    """
    Check if string contains substring.

    Inputs: value (string), search (substring to find)
    Outputs: result (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '')))
        search = str(inputs.get('search', ''))

        result = search in value
        return {'result': result, 'out': result}


class StringRegexFacet:
    """
    Match string against regex pattern.

    Inputs: value (string), pattern (regex)
    Outputs: result (boolean match), groups (captured groups)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '')))
        pattern = str(inputs.get('pattern', ''))

        try:
            match = re.search(pattern, value)
            if match:
                return {
                    'result': True,
                    'out': True,
                    'groups': list(match.groups()),
                    'match': match.group(0)
                }
            else:
                return {'result': False, 'out': False, 'groups': [], 'match': ''}
        except re.error as e:
            return {'result': False, 'out': False, 'error': str(e)}


# =============================================================================
# ARRAY/LIST FACETS
# =============================================================================

class ArrayGetFacet:
    """
    Get element from array by index.

    Inputs: array, index (default 0)
    Outputs: result (element), found (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        array = inputs.get('array', inputs.get('in', []))
        index = int(inputs.get('index', 0))

        if isinstance(array, str):
            try:
                array = json.loads(array)
            except json.JSONDecodeError:
                array = [array]

        if not isinstance(array, (list, tuple)):
            array = [array]

        if 0 <= index < len(array):
            return {'result': array[index], 'out': array[index], 'found': True}
        else:
            return {'result': None, 'out': None, 'found': False}


class ArrayJoinFacet:
    """
    Join array elements into string.

    Inputs: array
    Config: separator in prompt field (default: ', ')
    Outputs: result (joined string)
    """

    def __init__(self, facet_id: str, separator: str = ', '):
        self.facet_id = facet_id
        self.separator = separator

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        array = inputs.get('array', inputs.get('in', []))

        if isinstance(array, str):
            try:
                array = json.loads(array)
            except json.JSONDecodeError:
                array = [array]

        if not isinstance(array, (list, tuple)):
            array = [array]

        result = self.separator.join(str(item) for item in array)
        return {'result': result, 'out': result}


class ArrayLengthFacet:
    """
    Get length of array.

    Inputs: array
    Outputs: result (integer length)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        array = inputs.get('array', inputs.get('in', []))

        if isinstance(array, str):
            try:
                array = json.loads(array)
            except json.JSONDecodeError:
                result = len(array)  # String length
                return {'result': result, 'out': result}

        if not isinstance(array, (list, tuple)):
            result = 1
        else:
            result = len(array)

        return {'result': result, 'out': result}


class ArrayFirstFacet:
    """
    Get first element from array.

    Inputs: array
    Outputs: result (first element), empty (boolean if array was empty)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        array = inputs.get('array', inputs.get('in', []))

        if isinstance(array, str):
            try:
                array = json.loads(array)
            except json.JSONDecodeError:
                return {'result': array, 'out': array, 'empty': False}

        if not isinstance(array, (list, tuple)):
            return {'result': array, 'out': array, 'empty': False}

        if array:
            return {'result': array[0], 'out': array[0], 'empty': False}
        else:
            return {'result': None, 'out': None, 'empty': True}


class ArrayLastFacet:
    """
    Get last element from array.

    Inputs: array
    Outputs: result (last element), empty (boolean if array was empty)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        array = inputs.get('array', inputs.get('in', []))

        if isinstance(array, str):
            try:
                array = json.loads(array)
            except json.JSONDecodeError:
                return {'result': array, 'out': array, 'empty': False}

        if not isinstance(array, (list, tuple)):
            return {'result': array, 'out': array, 'empty': False}

        if array:
            return {'result': array[-1], 'out': array[-1], 'empty': False}
        else:
            return {'result': None, 'out': None, 'empty': True}


# =============================================================================
# CONTROL/DATA FLOW FACETS
# =============================================================================

class PassThroughFacet:
    """
    Pass input through unchanged. Useful for organizing connections.

    Inputs: in (any value)
    Outputs: out (same value)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = inputs.get('in', inputs.get('value', ''))
        return {'result': value, 'out': value}


class GateFacet:
    """
    Conditionally pass or block value based on gate condition.

    Inputs: value, gate (boolean - if true, pass value; if false, block)
    Outputs: result (value if gate open, None if blocked), passed (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = inputs.get('value', inputs.get('in', ''))
        gate = bool(inputs.get('gate', True))

        if gate:
            return {'result': value, 'out': value, 'passed': True}
        else:
            return {'result': None, 'out': None, 'passed': False}


class CounterFacet:
    """
    Increment counter on each execution. Stateful facet.

    Inputs: reset (optional, resets counter if true)
    Outputs: count (current count), out (count as string)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id
        self.count = 0

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        reset = bool(inputs.get('reset', False))

        if reset:
            self.count = 0
        else:
            self.count += 1

        return {'count': self.count, 'result': self.count, 'out': str(self.count)}


class JSONParseFacet:
    """
    Parse JSON string into object.

    Inputs: value (JSON string)
    Outputs: result (parsed object), error (if parsing failed)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = str(inputs.get('value', inputs.get('in', '{}')))

        try:
            result = json.loads(value)
            return {'result': result, 'out': result, 'success': True}
        except json.JSONDecodeError as e:
            return {'result': None, 'out': None, 'error': str(e), 'success': False}


class JSONStringifyFacet:
    """
    Convert object to JSON string.

    Inputs: value (any JSON-serializable object)
    Config: indent in prompt field (default: None for compact)
    Outputs: result (JSON string)
    """

    def __init__(self, facet_id: str, indent: Optional[int] = None):
        self.facet_id = facet_id
        self.indent = indent

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        value = inputs.get('value', inputs.get('in', {}))

        try:
            result = json.dumps(value, indent=self.indent, ensure_ascii=False)
            return {'result': result, 'out': result, 'success': True}
        except (TypeError, ValueError) as e:
            return {'result': '{}', 'out': '{}', 'error': str(e), 'success': False}


class GetPropertyFacet:
    """
    Get property from object/dict.

    Inputs: object (dict or JSON string), key (property name)
    Outputs: result (property value), found (boolean)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        obj = inputs.get('object', inputs.get('in', {}))
        key = str(inputs.get('key', ''))

        # Parse JSON if string
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except json.JSONDecodeError:
                obj = {}

        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            return {'result': value, 'out': value, 'found': True}
        else:
            return {'result': None, 'out': None, 'found': False}


class SetPropertyFacet:
    """
    Set property on object/dict.

    Inputs: object (dict or JSON string), key (property name), value
    Outputs: result (modified object)
    """

    def __init__(self, facet_id: str):
        self.facet_id = facet_id

    def process(self, inputs: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        obj = inputs.get('object', inputs.get('in', {}))
        key = str(inputs.get('key', ''))
        value = inputs.get('value', None)

        # Parse JSON if string
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except json.JSONDecodeError:
                obj = {}

        if not isinstance(obj, dict):
            obj = {}

        # Copy to avoid mutation
        result = dict(obj)
        result[key] = value

        return {'result': result, 'out': result}


# =============================================================================
# FACET REGISTRY
# =============================================================================

# Map facet type strings to classes for instantiation
UTILITY_FACET_TYPES = {
    # Math
    'MathAddFacet': MathAddFacet,
    'MathSubtractFacet': MathSubtractFacet,
    'MathMultiplyFacet': MathMultiplyFacet,
    'MathDivideFacet': MathDivideFacet,
    'MathMinFacet': MathMinFacet,
    'MathMaxFacet': MathMaxFacet,
    'MathClampFacet': MathClampFacet,
    'MathAbsFacet': MathAbsFacet,

    # Logic
    'LogicAndFacet': LogicAndFacet,
    'LogicOrFacet': LogicOrFacet,
    'LogicNotFacet': LogicNotFacet,
    'LogicCompareFacet': LogicCompareFacet,
    'LogicSwitchFacet': LogicSwitchFacet,

    # String
    'StringConcatFacet': StringConcatFacet,
    'StringSplitFacet': StringSplitFacet,
    'StringReplaceFacet': StringReplaceFacet,
    'StringFormatFacet': StringFormatFacet,
    'StringLengthFacet': StringLengthFacet,
    'StringContainsFacet': StringContainsFacet,
    'StringRegexFacet': StringRegexFacet,

    # Array
    'ArrayGetFacet': ArrayGetFacet,
    'ArrayJoinFacet': ArrayJoinFacet,
    'ArrayLengthFacet': ArrayLengthFacet,
    'ArrayFirstFacet': ArrayFirstFacet,
    'ArrayLastFacet': ArrayLastFacet,

    # Control/Data
    'PassThroughFacet': PassThroughFacet,
    'GateFacet': GateFacet,
    'CounterFacet': CounterFacet,
    'JSONParseFacet': JSONParseFacet,
    'JSONStringifyFacet': JSONStringifyFacet,
    'GetPropertyFacet': GetPropertyFacet,
    'SetPropertyFacet': SetPropertyFacet,
}


def create_utility_facet(facet_type: str, facet_id: str, config: Optional[Dict] = None):
    """
    Factory function to create a utility facet.

    Args:
        facet_type: Type name from UTILITY_FACET_TYPES
        facet_id: Unique facet identifier
        config: Optional configuration dict (operator, separator, etc.)

    Returns:
        Facet instance or None if type not found
    """
    cls = UTILITY_FACET_TYPES.get(facet_type)
    if cls is None:
        return None

    # Some facets take config in constructor
    if facet_type == 'LogicCompareFacet' and config:
        return cls(facet_id, operator=config.get('operator', '=='))
    elif facet_type in ('StringConcatFacet', 'ArrayJoinFacet') and config:
        return cls(facet_id, separator=config.get('separator', ''))
    elif facet_type == 'StringSplitFacet' and config:
        return cls(facet_id, delimiter=config.get('delimiter', ' '))
    elif facet_type == 'JSONStringifyFacet' and config:
        indent = config.get('indent')
        return cls(facet_id, indent=int(indent) if indent else None)
    else:
        return cls(facet_id)
