import pygame
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

def visualize_model(model: nn.Module, 
                    input_data: Optional[torch.Tensor] = None,
                    width: int = 1200, 
                    height: int = 800,
                    title: str = "Neural Network Visualization"):
    """
    Visualize a PyTorch model with fully-connected layers using Pygame.
    
    Args:
        model: PyTorch model with Linear layers
        input_data: Optional input tensor to show activations
        width: Window width
        height: Window height
        title: Window title
    """
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    
    # Extract linear layers and their weights
    layers_info = []
    activations = []
    
    # Get layer dimensions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers_info.append({
                'name': name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'weight': module.weight.detach().cpu().numpy(),
                'bias': module.bias.detach().cpu().numpy() if module.bias is not None else None
            })
    
    if not layers_info:
        print("No Linear layers found in model!")
        pygame.quit()
        return
    
    # Compute activations if input provided
    if input_data is not None:
        model.eval()
        with torch.no_grad():
            x = input_data.flatten()
            activations.append(x.cpu().numpy())
            
            for layer_info in layers_info:
                # Find the actual layer module
                for module in model.modules():
                    if isinstance(module, nn.Linear) and \
                       module.in_features == layer_info['in_features'] and \
                       module.out_features == layer_info['out_features']:
                        x = module(x)
                        activations.append(x.cpu().numpy())
                        break
    
    # Calculate layout
    layer_sizes = [layers_info[0]['in_features']] + [l['out_features'] for l in layers_info]
    max_neurons = max(layer_sizes)
    
    # Spacing calculations
    layer_spacing = width // (len(layer_sizes) + 1)
    max_neuron_spacing = (height - 100) / max(max_neurons, 1)
    neuron_radius = min(15, max_neuron_spacing / 3)
    
    # Calculate positions for each neuron
    neuron_positions = []
    for layer_idx, layer_size in enumerate(layer_sizes):
        x = layer_spacing * (layer_idx + 1)
        layer_height = layer_size * max_neuron_spacing
        start_y = (height - layer_height) / 2
        
        positions = []
        for neuron_idx in range(layer_size):
            y = start_y + neuron_idx * max_neuron_spacing + max_neuron_spacing / 2
            positions.append((x, y))
        neuron_positions.append(positions)
    
    def value_to_color(value: float, vmin: float = -1, vmax: float = 1) -> Tuple[int, int, int]:
        """Convert a value to a color (blue for negative, red for positive)."""
        norm = (value - vmin) / (vmax - vmin + 1e-8)
        norm = np.clip(norm, 0, 1)
        
        if norm < 0.5:
            # Blue to white
            t = norm * 2
            return (int(255 * t), int(255 * t), 255)
        else:
            # White to red
            t = (norm - 0.5) * 2
            return (255, int(255 * (1 - t)), int(255 * (1 - t)))
    
    running = True
    font = pygame.font.Font(None, 20)
    small_font = pygame.font.Font(None, 16)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        screen.fill((20, 20, 20))
        
        # Draw connections (weights)
        for layer_idx, layer_info in enumerate(layers_info):
            weights = layer_info['weight']
            w_min, w_max = weights.min(), weights.max()
            
            from_positions = neuron_positions[layer_idx]
            to_positions = neuron_positions[layer_idx + 1]
            
            # Draw only a subset of connections if too many
            max_connections = 1000
            total_connections = len(from_positions) * len(to_positions)
            
            if total_connections > max_connections:
                # Sample connections
                step = int(np.ceil(total_connections / max_connections))
            else:
                step = 1
            
            connection_count = 0
            for to_idx, to_pos in enumerate(to_positions):
                for from_idx, from_pos in enumerate(from_positions):
                    if connection_count % step == 0:
                        weight = weights[to_idx, from_idx]
                        color = value_to_color(weight, w_min, w_max)
                        alpha = min(255, int(abs(weight) / max(abs(w_min), abs(w_max)) * 150) + 50)
                        
                        # Draw line with transparency
                        surf = pygame.Surface((width, height), pygame.SRCALPHA)
                        pygame.draw.line(surf, (*color, alpha), from_pos, to_pos, 1)
                        screen.blit(surf, (0, 0))
                    connection_count += 1
        
        # Draw neurons
        for layer_idx, positions in enumerate(neuron_positions):
            for neuron_idx, pos in enumerate(positions):
                # Determine neuron color based on activation
                if layer_idx < len(activations):
                    activation = activations[layer_idx][neuron_idx]
                    act_min = activations[layer_idx].min()
                    act_max = activations[layer_idx].max()
                    neuron_color = value_to_color(activation, act_min, act_max)
                else:
                    neuron_color = (100, 100, 100)
                
                # Draw neuron
                pygame.draw.circle(screen, neuron_color, (int(pos[0]), int(pos[1])), int(neuron_radius))
                pygame.draw.circle(screen, (200, 200, 200), (int(pos[0]), int(pos[1])), int(neuron_radius), 2)
                
                # Draw activation value if available
                if layer_idx < len(activations) and len(positions) <= 20:
                    value = activations[layer_idx][neuron_idx]
                    value_text = small_font.render(f"{value:.2f}", True, (255, 255, 255))
                    screen.blit(value_text, (pos[0] + neuron_radius + 5, pos[1] - 8))
        
        # Draw layer labels
        for layer_idx, positions in enumerate(neuron_positions):
            if positions:
                x = positions[0][0]
                if layer_idx == 0:
                    label = f"Input ({layer_sizes[layer_idx]})"
                elif layer_idx == len(neuron_positions) - 1:
                    label = f"Output ({layer_sizes[layer_idx]})"
                else:
                    label = f"Hidden {layer_idx} ({layer_sizes[layer_idx]})"
                
                text = font.render(label, True, (255, 255, 255))
                screen.blit(text, (x - text.get_width() // 2, 20))
        
        # Draw legend
        legend_y = height - 60
        legend_text = font.render("Weight/Activation: ", True, (255, 255, 255))
        screen.blit(legend_text, (20, legend_y))
        
        gradient_width = 200
        gradient_start = 200
        for i in range(gradient_width):
            value = (i / gradient_width) * 2 - 1
            color = value_to_color(value)
            pygame.draw.line(screen, color, (gradient_start + i, legend_y), (gradient_start + i, legend_y + 20))
        
        neg_text = small_font.render("Negative", True, (255, 255, 255))
        pos_text = small_font.render("Positive", True, (255, 255, 255))
        screen.blit(neg_text, (gradient_start, legend_y + 25))
        screen.blit(pos_text, (gradient_start + gradient_width - 40, legend_y + 25))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

