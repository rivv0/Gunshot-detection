# Microphone array-based direction of    arrival for gunshot detection
Developing Classification and Localization Algorithm

## ARCHITECTURE

[architecture diagram](img/architecture_diagram.png "Title Text")

Bandpass filters(BPF), allows frequencies in the band of 3KHz to pass through while attenuating others outside the range.
❖The following custom IP will contain the following new additions-

- First In First Out(FIFO) approach for temporary data storage at different rate of time ML model trained for Sound classification. And to estimate the type of gun.
- Localization algorithm approaches
  -Hexagonal Arrangement-of microphones such that each detects sound on different time intervals TDOA(Time Difference of Arrival)is calculated, forming hyperbola which intersects with each other pinpointing gunshot location , 99.99percent accuracy.[d12=v*ΔT12]
  [√(x−x1 )2+(y−y1 )2+(z−z1 )2−√(x−x2 )2+(y−y2 )2+(z−z2 )2=v*ΔT12]
- Calculate delta,i.e difference between source and the microphones arranged in Linearfashion,the six outputs can then be compared.


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
