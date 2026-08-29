# Vaani Hardware

## hardware overview

Vaani needs a small and efficient hardware platform capable of continuously capturing audio and running the voice activation pipeline locally. The final hardware will be selected based on actual testing rather than choosing components only from their specifications.

The hardware should provide enough processing capability for the selected ML model while keeping memory usage, power consumption, cost and overall size reasonable.

## main hardware blocks

The planned hardware can be divided into a few main blocks:

```text
Microphone
    ↓
Audio Interface
    ↓
Edge Processor
    ↓
Trigger Output

Power Supply
    ↓
All Hardware Blocks
```

The final design may include additional components depending on the selected platform and prototype requirements.

## microcontroller or edge processor

The main processor is responsible for audio processing, feature extraction, ML inference and system control.

The selected processor should be evaluated using factors such as:

- Processing capability
- RAM
- Flash or available storage
- Supported audio interfaces
- ML inference capability
- Power consumption
- Development support
- Cost
- Physical size

The final processor will be selected after comparing suitable options and testing the most practical candidates.

## microphone

The microphone is the primary input device for Vaani.

The microphone selection should consider:

- Audio quality
- Sampling support
- Signal to noise performance
- Interface
- Power consumption
- Physical size
- Availability
- Compatibility with the selected processor

A digital microphone may simplify integration, but the final choice will depend on the complete audio pipeline.

## power system

Since Vaani may need to listen continuously, power efficiency is an important part of the hardware design.

The power system will eventually include the required regulation and protection components for the selected hardware. If the final prototype is battery powered, battery capacity and expected operating time will also be evaluated.

Power measurements will be taken during different operating states such as idle listening, audio processing and inference.

## status and user interface

The prototype may include simple indicators or controls to make the device easier to test and demonstrate.

Possible components include:

- Status LED
- Push button
- Buzzer
- Display, if required

These are not core requirements and will only be added if they provide a useful purpose.

## communication

The main voice activation process is intended to run locally. Communication interfaces may still be used for development, configuration, debugging or triggering external devices.

The final communication method will depend on the selected processor and the requirements of the prototype.

## hardware prototype

The first prototype will likely use development boards and modules so that the audio pipeline and embedded inference can be tested quickly.

Once the architecture is stable, the components can be integrated into a cleaner hardware design and, if useful, a custom PCB.

The prototype enclosure will also be designed around the actual hardware rather than being finalized before the internal components are known.

## hardware selection process

Hardware will be selected using a practical comparison rather than focusing on a single specification.

The main factors will be:

| Factor | Why it matters |
|---|---|
| Processing | Determines how quickly the pipeline can run |
| RAM | Needed for audio buffers and ML inference |
| Storage | Needed for firmware and model |
| Audio interface | Determines microphone integration |
| Power | Important for continuous listening |
| Size | Affects the final Vaani enclosure |
| Cost | Important for prototype practicality |
| Availability | Avoids delays during development |
| Software support | Makes development and debugging easier |

## bill of materials

The final components and their costs will be maintained in:

`hardware/bom/bom.csv`

The BOM will be updated whenever a component is selected, replaced or added.

## schematics

Circuit schematics will be stored in:

`hardware/schematics/`

Different versions can be maintained as the design evolves.

For example:

```text
vaani-v1
vaani-v2
```

The reason for major hardware changes will be documented in the project documentation.

## PCB

If a custom PCB becomes useful after the prototype stage, its design files and manufacturing files will be stored in:

`hardware/pcb/`

The PCB will only be designed after the main hardware architecture has been tested.

## wiring

Early prototypes may use development boards, jumper wires and breadboards.

Wiring diagrams and pin assignments will be stored in:

`hardware/wiring/`

This should make it possible to reproduce the prototype without guessing connections.

## enclosure

The physical Vaani enclosure will be designed after the internal hardware is finalized.

The enclosure should provide:

- Microphone opening
- Access to required controls
- Status indication visibility
- Proper mounting of internal components
- Protection for the electronics
- A compact and practical form

CAD files and final prototype renders will be stored in:

`hardware/enclosure/`

## hardware development approach

The hardware development will happen in stages.

First, individual components will be tested separately. Then the microphone, processor and power system will be combined to create the first working prototype. After the embedded voice activation pipeline is stable, the hardware can be made more compact and integrated.

This avoids spending time and money on a final PCB or enclosure before knowing whether the underlying design actually works.

## current hardware status

The hardware has not been finalized yet. Component selection will be based on the requirements defined for Vaani and the results of the initial software and hardware experiments.
