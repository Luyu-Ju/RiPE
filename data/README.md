# Spatial input data

The simulation expects the following project-relative layout:

```text
data/
|-- Building/
|   `-- Buildings.*
|-- Departure/
|   `-- Departure.*
|-- GuideLine/
|   |-- GuidePoint1.*
|   |-- GuidePoint2.*
|   |-- GuidePoint3.*
|   |-- GuidePoint4.*
|   |-- GuidePoint5.*
|   `-- GuidePoint6.*
|-- Landslide/
|   |-- Arrow.*
|   |-- Flow direction.*
|   `-- Landslide.*
|-- Road/
|   |-- Boundary_Graph.*
|   |-- BoundaryObstacle.tif
|   |-- Road_Center.tif
|   `-- RoadPolygon.tif
`-- Shelter/
    |-- Shelter.tif
    |-- Sign.*
    `-- SignIn.*
```

For an ESRI Shapefile, all associated components such as `.shp`, `.shx`,
`.dbf`, `.prj`, and `.cpg` should remain together. Do not rename one component
without renaming all components that share its basename.

The simulation assumes that all spatial layers use compatible coordinate
reference systems and map units. The software license at the repository root
does not automatically apply to these datasets. Confirm that the source terms
permit publication and redistribution before making the GitHub repository
public.

