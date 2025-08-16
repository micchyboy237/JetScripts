from jet.logger import CustomLogger
import os
import shutil
import { AddComponentDropdown } from "../../shared";


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# AddComponentDropdown Usage Examples

The `AddComponentDropdown` component is a reusable dropdown that allows users to add components to a gallery. It supports all component types (teams, agents, models, tools, workbenches, terminations).

## Basic Usage
"""
logger.info("# AddComponentDropdown Usage Examples")


<AddComponentDropdown
  componentType="workbench"
  gallery={selectedGallery}
  onComponentAdded={handleComponentAdded}
/>;

"""
## Advanced Usage with Filtering (MCP Workbenches)
"""
logger.info("## Advanced Usage with Filtering (MCP Workbenches)")

<AddComponentDropdown
  componentType="workbench"
  gallery={selectedGallery}
  onComponentAdded={handleComponentAdded}
  size="small"
  type="text"
  buttonText="+"
  showChevron={false}
  templateFilter={(template) =>
    template.label.toLowerCase().includes("mcp") ||
    template.description.toLowerCase().includes("mcp")
  }
/>

"""
## Props

- `componentType`: The type of component to add (team, agent, model, tool, workbench, termination)
- `gallery`: The gallery to add the component to
- `onComponentAdded`: Callback when a component is added
- `disabled`: Whether the dropdown is disabled
- `showIcon`: Whether to show the plus icon
- `showChevron`: Whether to show the chevron down icon
- `size`: Button size
- `type`: Button type
- `className`: Additional CSS classes
- `buttonText`: Custom button text
- `templateFilter`: Optional filter function for templates

## Handler Signature
"""
logger.info("## Props")

const handleComponentAdded = (
  component: Component<ComponentConfig>,
  category: CategoryKey
) => {
  // Handle the added component
  // Update your gallery/state here
};

"""
## Benefits

1. **Reusability**: Use the same component across different views
2. **Consistency**: Same UI/UX everywhere
3. **Maintainability**: Single source of truth for component addition logic
4. **Flexibility**: Configurable with props and filters
5. **Type Safety**: Fully typed with TypeScript
"""
logger.info("## Benefits")

logger.info("\n\n[DONE]", bright=True)