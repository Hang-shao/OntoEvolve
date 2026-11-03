import java.io.File;
import java.util.Set;

import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.io.FileDocumentSource;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyLoaderConfiguration;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.semanticweb.owlapi.model.parameters.Imports;
import org.semanticweb.owlapi.rdf.rdfxml.parser.RDFXMLParser;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

/**
 * HermiT consistency check entry point (OWL 2 DL).
 *
 * Why not the bundled `org.semanticweb.HermiT.cli.CommandLine`:
 *   OWLAPI picks parsers through SPI, and a single failing provider makes
 *   its Injector hand out a null parser factory (NPE) instead of skipping it.
 *   Here the RDF/XML parser is instantiated directly, so no SPI is involved.
 *
 * Usage:  java HermiTCheck <owl-file>
 * Output: CONSISTENT=true|false, then one UNSAT:<class-iri> line per
 *         unsatisfiable class (only when inconsistent).
 */
public class HermiTCheck {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("CONSISTENT=false");
            System.out.println("ERROR=no-input");
            System.exit(2);
        }
        File file = new File(args[0]);
        OWLOntologyManager man = OWLManager.createOWLOntologyManager();
        OWLOntology ont = man.createOntology();
        try {
            RDFXMLParser parser = new RDFXMLParser();
            parser.parse(new FileDocumentSource(file), ont,
                    new OWLOntologyLoaderConfiguration());
        } catch (Exception e) {
            System.out.println("CONSISTENT=false");
            System.out.println("ERROR=parse:" + e.getMessage());
            System.exit(3);
        }
        OWLReasoner reasoner = new Reasoner.ReasonerFactory().createReasoner(ont);
        boolean consistent;
        try {
            consistent = reasoner.isConsistent();
        } catch (Exception e) {
            System.out.println("CONSISTENT=false");
            System.out.println("ERROR=reason:" + e);
            System.exit(4);
            return;
        }
        System.out.println("CONSISTENT=" + consistent);
        if (!consistent) {
            // 本体不一致时 OWLAPI 拒绝对其做任何 satisfiability 查询
            // （InconsistentOntologyException），因此这里另建一份只含 TBox 的
            // 副本，用来定位真正不可满足的类（诊断信息）。
            try {
                OWLOntology tbox = man.createOntology();
                man.addAxioms(tbox, ont.getTBoxAxioms(Imports.EXCLUDED));
                OWLReasoner r2 = new Reasoner.ReasonerFactory().createReasoner(tbox);
                for (OWLClass c : tbox.getClassesInSignature()) {
                    if (!r2.isSatisfiable(c)) {
                        System.out.println("UNSAT:" + c.getIRI());
                    }
                }
                r2.dispose();
            } catch (Throwable t) {
                System.out.println("UNSAT-ERROR:" + t);
            }
        }
        reasoner.dispose();
    }
}
