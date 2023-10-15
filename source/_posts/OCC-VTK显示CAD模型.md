---
title: OCC+VTK显示CAD模型
date: 2023-10-15 16:53:58
categories:
- 前后处理
tags:
- OCC
- VTK
- 可视化
---

## 前言

早些时候，这篇文章已经被我发布在了知乎上：“[VTK+OCC显示CAD模型](https://zhuanlan.zhihu.com/p/455592800)”，今天将文章重新编辑，发布在Github上，主要是为了测试Hexo的用法。

## 测试环境

* 系统：Win11
* IDE：Visual Studio Community 2019
* VTK：VTK 9.2
* OCC：OCCT 7.6

## 正文

VTK是一款十分优秀的可视化套件，开源且功能强大，基本上可以满足有限元领域的全部可视化需求。遗憾的是，VTK不支持CAD模型（如igs、stp格式的模型）的显示。

在网上搜索后可以发现，在不花钱的情况下，想要显示和处理CAD模型，基本上都得使用OpenCasCade，即OCC。OCC有自己的可视化系统，也可以集成在Qt中。但对我而已，OCC自己的可视化系统还是太复杂了。

好在OCC在6.8版本开发了VIS（VTK Integration Services）功能，之后的版本就可以使用VTK进行模型的可视化了。

为了使用VIS功能，编译OCC的时候需要选择USE_VTK的选项，编译完成后，将生成TKIVtk、TKIVtkDraw的动态库和静态库。如果编译路径下有这两个库，说明VIS的功能是编译成功了。

编写一个最小案例看看显示效果。

CMakeLists.txt文件：

``` CMake
cmake_minimum_required(VERSION 3.15)

project(occvtk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VTK REQUIRED)
find_package(OpenCASCADE REQUIRED)
include_directories(${OpenCASCADE_INCLUDE_DIR})
link_directories(${OpenCASCADE_LIBRARY_DIR})

add_executable(test test.cpp)
target_link_libraries(test
    ${VTK_LIBRARIES}
    ${OpenCASCADE_LIBRARIES}
)
vtk_module_autoinit(
    TARGETS test
    MODULES ${VTK_LIBRARIES}
)
```

在此CMakeLists文件中，没有写死VTK和OCC的路径，而是使用find_package命令查找。如果没有找到，可以在命令行执行cmake命令时使用-D参数将相关路径传入:


``` bash
cmake .. -DVTK_DIR:PATH=/home/me/vtk_build -DVTK_DIR -DOpenCASCADE_DIR:PATH=/home/me/occ_build
```

test.cpp文件:

``` C++
#include <STEPControl_Reader.hxx>
#include <Standard_Integer.hxx>
#include <TopoDS_Shape.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IFSelect_PrintCount.hxx>
#include <IVtkTools_ShapeDataSource.hxx>
#include <IVtkOCC_ShapeMesher.hxx>
#include <IVtkTools_DisplayModeFilter.hxx>
#include <vtkType.h>
#include <vtkAutoInit.h>
#include <vtkRenderWindow.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyDataMapper.h>
#include <vtkInteractorStyleTrackballCamera.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle)

int main()
{
    STEPControl_Reader reader;
    IFSelect_ReturnStatus stat = reader.ReadFile("assembly_solid.stp");
    IFSelect_PrintCount mode = IFSelect_CountByItem;
    Standard_Integer NbRoots = reader.NbRootsForTransfer();
    Standard_Integer num = reader.TransferRoots();
    Standard_Integer NbTrans = reader.TransferRoots();
    TopoDS_Shape result = reader.OneShape();
    // TopoDS_Shape shape = reader.Shape();

    vtkNew<IVtkTools_ShapeDataSource> occSource;
    //occSource->SetShape(new IVtkOCC_Shape(shape));
    occSource->SetShape(new IVtkOCC_Shape(result));

    vtkNew<IVtkTools_DisplayModeFilter> filter;
    filter->AddInputConnection(occSource->GetOutputPort());
    filter->SetDisplayMode(DM_Shading);

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(filter->GetOutputPort());

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);

    vtkNew<vtkRenderer> ren;
    ren->AddActor(actor);

    vtkNew<vtkRenderWindow> renWin;
    renWin->AddRenderer(ren);
    renWin->SetSize(960, 800);

    vtkNew<vtkInteractorStyleTrackballCamera> istyle;
    vtkNew<vtkRenderWindowInteractor> iren;

    iren->SetRenderWindow(renWin);
    iren->SetInteractorStyle(istyle);

    renWin->Render();
    iren->Start();

    return 0;
}
```

此测试代码将模型文件“assembly_solid.stp”写死在了源代码中，编译完成后，需要确保可执行文件可以找到模型文件。

显示效果如下：

<img src="{% asset_path assembly.jpg %}" />
